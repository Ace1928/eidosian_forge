import types as pytypes  # avoid confusion with numba.types
import sys, math
import os
import textwrap
import copy
import inspect
import linecache
from functools import reduce
from collections import defaultdict, OrderedDict, namedtuple
from contextlib import contextmanager
import operator
from dataclasses import make_dataclass
import warnings
from llvmlite import ir as lir
from numba.core.imputils import impl_ret_untracked
import numba.core.ir
from numba.core import types, typing, utils, errors, ir, analysis, postproc, rewrites, typeinfer, config, ir_utils
from numba import prange, pndindex
from numba.np.npdatetime_helpers import datetime_minimum, datetime_maximum
from numba.np.numpy_support import as_dtype, numpy_version
from numba.core.typing.templates import infer_global, AbstractTemplate
from numba.stencils.stencilparfor import StencilPass
from numba.core.extending import register_jitable, lower_builtin
from numba.core.ir_utils import (
from numba.core.analysis import (compute_use_defs, compute_live_map,
from numba.core.controlflow import CFGraph
from numba.core.typing import npydecl, signature
from numba.core.types.functions import Function
from numba.parfors.array_analysis import (random_int_args, random_1arg_size,
from numba.core.extending import overload
import copy
import numpy
import numpy as np
from numba.parfors import array_analysis
import numba.cpython.builtins
from numba.stencils import stencilparfor
class ParforDiagnostics(object):
    """Holds parfor diagnostic info, this is accumulated throughout the
    PreParforPass and ParforPass, also in the closure inlining!
    """

    def __init__(self):
        self.func = None
        self.replaced_fns = dict()
        self.internal_name = '__numba_parfor_gufunc'
        self.fusion_info = defaultdict(list)
        self.nested_fusion_info = defaultdict(list)
        self.fusion_reports = []
        self.hoist_info = {}
        self.has_setup = False

    def setup(self, func_ir, fusion_enabled):
        self.func_ir = func_ir
        self.name = self.func_ir.func_id.func_qualname
        self.line = self.func_ir.loc
        self.fusion_enabled = fusion_enabled
        if self.internal_name in self.name:
            self.purpose = 'Internal parallel function'
        else:
            self.purpose = 'Function %s, %s' % (self.name, self.line)
        self.initial_parfors = self.get_parfors()
        self.has_setup = True

    @property
    def has_setup(self):
        return self._has_setup

    @has_setup.setter
    def has_setup(self, state):
        self._has_setup = state

    def count_parfors(self, blocks=None):
        return len(self.get_parfors())

    def _get_nested_parfors(self, parfor, parfors_list):
        blocks = wrap_parfor_blocks(parfor)
        self._get_parfors(blocks, parfors_list)
        unwrap_parfor_blocks(parfor)

    def _get_parfors(self, blocks, parfors_list):
        for label, blk in blocks.items():
            for stmt in blk.body:
                if isinstance(stmt, Parfor):
                    parfors_list.append(stmt)
                    self._get_nested_parfors(stmt, parfors_list)

    def get_parfors(self):
        parfors_list = []
        self._get_parfors(self.func_ir.blocks, parfors_list)
        return parfors_list

    def hoisted_allocations(self):
        allocs = []
        for pf_id, data in self.hoist_info.items():
            stmt = data.get('hoisted', [])
            for inst in stmt:
                if isinstance(inst.value, ir.Expr):
                    if inst.value.op == 'call':
                        call = guard(find_callname, self.func_ir, inst.value)
                        if call is not None and call == ('empty', 'numpy'):
                            allocs.append(inst)
        return allocs

    def compute_graph_info(self, _a):
        """
        compute adjacency list of the fused loops
        and find the roots in of the lists
        """
        a = copy.deepcopy(_a)
        if a == {}:
            return ([], set())
        vtx = set()
        for v in a.values():
            for x in v:
                vtx.add(x)
        potential_roots = set(a.keys())
        roots = potential_roots - vtx
        if roots is None:
            roots = set()
        not_roots = set()
        for x in range(max(set(a.keys()).union(vtx)) + 1):
            val = a.get(x)
            if val is not None:
                a[x] = val
            elif val == []:
                not_roots.add(x)
            else:
                a[x] = []
        l = []
        for x in sorted(a.keys()):
            l.append(a[x])
        return (l, roots)

    def get_stats(self, fadj, nadj, root):
        """
        Computes the number of fused and serialized loops
        based on a fusion adjacency list `fadj` and a nested
        parfors adjacency list `nadj` for the root, `root`
        """

        def count_root(fadj, nadj, root, nfused, nserial):
            for k in nadj[root]:
                nserial += 1
                if nadj[k] == []:
                    nfused += len(fadj[k])
                else:
                    nf, ns = count_root(fadj, nadj, k, nfused, nserial)
                    nfused += nf
                    nserial = ns
            return (nfused, nserial)
        nfused, nserial = count_root(fadj, nadj, root, 0, 0)
        return (nfused, nserial)

    def reachable_nodes(self, adj, root):
        """
        returns a list of nodes reachable in an adjacency list from a
        specified root
        """
        fusers = []
        fusers.extend(adj[root])
        for k in adj[root]:
            if adj[k] != []:
                fusers.extend(self.reachable_nodes(adj, k))
        return fusers

    def sort_pf_by_line(self, pf_id, parfors_simple):
        """
        pd_id - the parfors id
        parfors_simple - the simple parfors map
        """
        pf = parfors_simple[pf_id][0]
        pattern = pf.patterns[0]
        line = max(0, pf.loc.line - 1)
        filename = self.func_ir.loc.filename
        nadj, nroots = self.compute_graph_info(self.nested_fusion_info)
        fadj, froots = self.compute_graph_info(self.fusion_info)
        graphs = [nadj, fadj]
        if isinstance(pattern, tuple):
            if pattern[1] == 'internal':
                reported_loc = pattern[2][1]
                if reported_loc.filename == filename:
                    return max(0, reported_loc.line - 1)
                else:
                    tmp = []
                    for adj in graphs:
                        if adj:
                            for k in adj[pf_id]:
                                tmp.append(self.sort_pf_by_line(k, parfors_simple))
                            if tmp:
                                return max(0, min(tmp) - 1)
                    for blk in pf.loop_body.values():
                        for stmt in blk.body:
                            if stmt.loc.filename == filename:
                                return max(0, stmt.loc.line - 1)
                    for blk in self.func_ir.blocks.values():
                        try:
                            idx = blk.body.index(pf)
                            for i in range(idx - 1, 0, -1):
                                stmt = blk.body[i]
                                if not isinstance(stmt, Parfor):
                                    line = max(0, stmt.loc.line - 1)
                                    break
                        except ValueError:
                            pass
        return line

    def get_parfors_simple(self, print_loop_search):
        parfors_simple = dict()
        for pf in sorted(self.initial_parfors, key=lambda x: x.loc.line):
            r_pattern = pf.patterns[0]
            pattern = pf.patterns[0]
            loc = pf.loc
            if isinstance(pattern, tuple):
                if pattern[0] == 'prange':
                    if pattern[1] == 'internal':
                        replfn = '.'.join(reversed(list(pattern[2][0])))
                        loc = pattern[2][1]
                        r_pattern = '%s %s' % (replfn, '(internal parallel version)')
                    elif pattern[1] == 'user':
                        r_pattern = 'user defined prange'
                    elif pattern[1] == 'pndindex':
                        r_pattern = 'internal pndindex'
                    else:
                        assert 0
            fmt = 'Parallel for-loop #%s: is produced from %s:\n    %s\n \n'
            if print_loop_search:
                print_wrapped(fmt % (pf.id, loc, r_pattern))
            parfors_simple[pf.id] = (pf, loc, r_pattern)
        return parfors_simple

    def get_all_lines(self, parfors_simple):
        fadj, froots = self.compute_graph_info(self.fusion_info)
        nadj, _nroots = self.compute_graph_info(self.nested_fusion_info)
        if len(fadj) > len(nadj):
            lim = len(fadj)
            tmp = nadj
        else:
            lim = len(nadj)
            tmp = fadj
        for x in range(len(tmp), lim):
            tmp.append([])
        nroots = set()
        if _nroots:
            for r in _nroots:
                if nadj[r] != []:
                    nroots.add(r)
        all_roots = froots ^ nroots
        froots_lines = {}
        for x in froots:
            line = self.sort_pf_by_line(x, parfors_simple)
            froots_lines[line] = ('fuse', x, fadj)
        nroots_lines = {}
        for x in nroots:
            line = self.sort_pf_by_line(x, parfors_simple)
            nroots_lines[line] = ('nest', x, nadj)
        all_lines = froots_lines.copy()
        all_lines.update(nroots_lines)
        return all_lines

    def source_listing(self, parfors_simple, purpose_str):
        filename = self.func_ir.loc.filename
        count = self.count_parfors()
        func_name = self.func_ir.func_id.func
        try:
            lines = inspect.getsource(func_name).splitlines()
        except OSError:
            lines = None
        if lines and parfors_simple:
            src_width = max([len(x) for x in lines])
            map_line_to_pf = defaultdict(list)
            for k, v in parfors_simple.items():
                if parfors_simple[k][1].filename == filename:
                    match_line = self.sort_pf_by_line(k, parfors_simple)
                    map_line_to_pf[match_line].append(str(k))
            max_pf_per_line = max([1] + [len(x) for x in map_line_to_pf.values()])
            width = src_width + (1 + max_pf_per_line * (len(str(count)) + 2))
            newlines = []
            newlines.append('\n')
            newlines.append('Parallel loop listing for %s' % purpose_str)
            newlines.append(width * '-' + '|loop #ID')
            fmt = '{0:{1}}| {2}'
            lstart = max(0, self.func_ir.loc.line - 1)
            for no, line in enumerate(lines, lstart):
                pf_ids = map_line_to_pf.get(no, None)
                if pf_ids is not None:
                    pfstr = '#' + ', '.join(pf_ids)
                else:
                    pfstr = ''
                stripped = line.strip('\n')
                srclen = len(stripped)
                if pf_ids:
                    l = fmt.format(width * '-', width, pfstr)
                else:
                    l = fmt.format(width * ' ', width, pfstr)
                newlines.append(stripped + l[srclen:])
            print('\n'.join(newlines))
        else:
            print('No source available')

    def print_unoptimised(self, lines):
        sword = '+--'
        fac = len(sword)
        fadj, froots = self.compute_graph_info(self.fusion_info)
        nadj, _nroots = self.compute_graph_info(self.nested_fusion_info)
        if len(fadj) > len(nadj):
            lim = len(fadj)
            tmp = nadj
        else:
            lim = len(nadj)
            tmp = fadj
        for x in range(len(tmp), lim):
            tmp.append([])

        def print_nest(fadj_, nadj_, theroot, reported, region_id):

            def print_g(fadj_, nadj_, nroot, depth):
                print_wrapped(fac * depth * ' ' + '%s%s %s' % (sword, nroot, '(parallel)'))
                for k in nadj_[nroot]:
                    if nadj_[k] == []:
                        msg = []
                        msg.append(fac * (depth + 1) * ' ' + '%s%s %s' % (sword, k, '(parallel)'))
                        if fadj_[k] != [] and k not in reported:
                            fused = self.reachable_nodes(fadj_, k)
                            for i in fused:
                                msg.append(fac * (depth + 1) * ' ' + '%s%s %s' % (sword, i, '(parallel)'))
                        reported.append(k)
                        print_wrapped('\n'.join(msg))
                    else:
                        print_g(fadj_, nadj_, k, depth + 1)
            if nadj_[theroot] != []:
                print_wrapped('Parallel region %s:' % region_id)
                print_g(fadj_, nadj_, theroot, 0)
                print('\n')
                region_id = region_id + 1
            return region_id

        def print_fuse(ty, pf_id, adj, depth, region_id):
            msg = []
            print_wrapped('Parallel region %s:' % region_id)
            msg.append(fac * depth * ' ' + '%s%s %s' % (sword, pf_id, '(parallel)'))
            if adj[pf_id] != []:
                fused = sorted(self.reachable_nodes(adj, pf_id))
                for k in fused:
                    msg.append(fac * depth * ' ' + '%s%s %s' % (sword, k, '(parallel)'))
            region_id = region_id + 1
            print_wrapped('\n'.join(msg))
            print('\n')
            return region_id
        region_id = 0
        reported = []
        for line, info in sorted(lines.items()):
            opt_ty, pf_id, adj = info
            if opt_ty == 'fuse':
                if pf_id not in reported:
                    region_id = print_fuse('f', pf_id, adj, 0, region_id)
            elif opt_ty == 'nest':
                region_id = print_nest(fadj, nadj, pf_id, reported, region_id)
            else:
                assert 0

    def print_optimised(self, lines):
        sword = '+--'
        fac = len(sword)
        fadj, froots = self.compute_graph_info(self.fusion_info)
        nadj, _nroots = self.compute_graph_info(self.nested_fusion_info)
        if len(fadj) > len(nadj):
            lim = len(fadj)
            tmp = nadj
        else:
            lim = len(nadj)
            tmp = fadj
        for x in range(len(tmp), lim):
            tmp.append([])
        summary = dict()

        def print_nest(fadj_, nadj_, theroot, reported, region_id):

            def print_g(fadj_, nadj_, nroot, depth):
                for k in nadj_[nroot]:
                    msg = fac * depth * ' ' + '%s%s %s' % (sword, k, '(serial')
                    if nadj_[k] == []:
                        fused = []
                        if fadj_[k] != [] and k not in reported:
                            fused = sorted(self.reachable_nodes(fadj_, k))
                            msg += ', fused with loop(s): '
                            msg += ', '.join([str(x) for x in fused])
                        msg += ')'
                        reported.append(k)
                        print_wrapped(msg)
                        summary[region_id]['fused'] += len(fused)
                    else:
                        print_wrapped(msg + ')')
                        print_g(fadj_, nadj_, k, depth + 1)
                    summary[region_id]['serialized'] += 1
            if nadj_[theroot] != []:
                print_wrapped('Parallel region %s:' % region_id)
                print_wrapped('%s%s %s' % (sword, theroot, '(parallel)'))
                summary[region_id] = {'root': theroot, 'fused': 0, 'serialized': 0}
                print_g(fadj_, nadj_, theroot, 1)
                print('\n')
                region_id = region_id + 1
            return region_id

        def print_fuse(ty, pf_id, adj, depth, region_id):
            print_wrapped('Parallel region %s:' % region_id)
            msg = fac * depth * ' ' + '%s%s %s' % (sword, pf_id, '(parallel')
            fused = []
            if adj[pf_id] != []:
                fused = sorted(self.reachable_nodes(adj, pf_id))
                msg += ', fused with loop(s): '
                msg += ', '.join([str(x) for x in fused])
            summary[region_id] = {'root': pf_id, 'fused': len(fused), 'serialized': 0}
            msg += ')'
            print_wrapped(msg)
            print('\n')
            region_id = region_id + 1
            return region_id
        region_id = 0
        reported = []
        for line, info in sorted(lines.items()):
            opt_ty, pf_id, adj = info
            if opt_ty == 'fuse':
                if pf_id not in reported:
                    region_id = print_fuse('f', pf_id, adj, 0, region_id)
            elif opt_ty == 'nest':
                region_id = print_nest(fadj, nadj, pf_id, reported, region_id)
            else:
                assert 0
        if summary:
            for k, v in sorted(summary.items()):
                msg = '\n \nParallel region %s (loop #%s) had %s loop(s) fused'
                root = v['root']
                fused = v['fused']
                serialized = v['serialized']
                if serialized != 0:
                    msg += ' and %s loop(s) serialized as part of the larger parallel loop (#%s).'
                    print_wrapped(msg % (k, root, fused, serialized, root))
                else:
                    msg += '.'
                    print_wrapped(msg % (k, root, fused))
        else:
            print_wrapped('Parallel structure is already optimal.')

    def allocation_hoist(self):
        found = False
        print('Allocation hoisting:')
        for pf_id, data in self.hoist_info.items():
            stmt = data.get('hoisted', [])
            for inst in stmt:
                if isinstance(inst.value, ir.Expr):
                    try:
                        attr = inst.value.attr
                        if attr == 'empty':
                            msg = 'The memory allocation derived from the instruction at %s is hoisted out of the parallel loop labelled #%s (it will be performed before the loop is executed and reused inside the loop):'
                            loc = inst.loc
                            print_wrapped(msg % (loc, pf_id))
                            try:
                                path = os.path.relpath(loc.filename)
                            except ValueError:
                                path = os.path.abspath(loc.filename)
                            lines = linecache.getlines(path)
                            if lines and loc.line:
                                print_wrapped('   Allocation:: ' + lines[0 if loc.line < 2 else loc.line - 1].strip())
                            print_wrapped('    - numpy.empty() is used for the allocation.\n')
                            found = True
                    except (KeyError, AttributeError):
                        pass
        if not found:
            print_wrapped('No allocation hoisting found')

    def instruction_hoist(self):
        print('')
        print('Instruction hoisting:')
        hoist_info_printed = False
        if self.hoist_info:
            for pf_id, data in self.hoist_info.items():
                hoisted = data.get('hoisted', None)
                not_hoisted = data.get('not_hoisted', None)
                if not hoisted and (not not_hoisted):
                    print('loop #%s has nothing to hoist.' % pf_id)
                    continue
                print('loop #%s:' % pf_id)
                if hoisted:
                    print('  Has the following hoisted:')
                    [print('    %s' % y) for y in hoisted]
                    hoist_info_printed = True
                if not_hoisted:
                    print('  Failed to hoist the following:')
                    [print('    %s: %s' % (y, x)) for x, y in not_hoisted]
                    hoist_info_printed = True
        if not hoist_info_printed:
            print_wrapped('No instruction hoisting found')
        print_wrapped(80 * '-')

    def dump(self, level=1):
        if not self.has_setup:
            raise RuntimeError('self.setup has not been called')
        name = self.func_ir.func_id.func_qualname
        line = self.func_ir.loc
        if self.internal_name in name:
            purpose_str = 'Internal parallel functions '
            purpose = 'internal'
        else:
            purpose_str = ' Function %s, %s ' % (name, line)
            purpose = 'user'
        print_loop_search = False
        print_source_listing = False
        print_fusion_search = False
        print_fusion_summary = False
        print_loopnest_rewrite = False
        print_pre_optimised = False
        print_post_optimised = False
        print_allocation_hoist = False
        print_instruction_hoist = False
        print_internal = False
        if level in (1, 2, 3, 4):
            print_source_listing = True
            print_post_optimised = True
        else:
            raise ValueError('Report level unknown, should be one of 1, 2, 3, 4')
        if level in (2, 3, 4):
            print_pre_optimised = True
        if level in (3, 4):
            print_allocation_hoist = True
        if level == 3:
            print_fusion_summary = True
            print_loopnest_rewrite = True
        if level == 4:
            print_fusion_search = True
            print_instruction_hoist = True
            print_internal = True
        if purpose == 'internal' and (not print_internal):
            return
        print_wrapped('\n ')
        print_wrapped(_termwidth * '=')
        print_wrapped((' Parallel Accelerator Optimizing: %s ' % purpose_str).center(_termwidth, '='))
        print_wrapped(_termwidth * '=')
        print_wrapped('')
        if print_loop_search:
            print_wrapped('Looking for parallel loops'.center(_termwidth, '-'))
        parfors_simple = self.get_parfors_simple(print_loop_search)
        count = self.count_parfors()
        if print_loop_search:
            print_wrapped('\nFound %s parallel loops.' % count)
            print_wrapped('-' * _termwidth)
        filename = self.func_ir.loc.filename
        try:
            path = os.path.relpath(filename)
        except ValueError:
            path = os.path.abspath(filename)
        if print_source_listing:
            self.source_listing(parfors_simple, purpose_str)
        sword = '+--'
        parfors = self.get_parfors()
        parfor_ids = [x.id for x in parfors]
        n_parfors = len(parfor_ids)
        if print_fusion_search or print_fusion_summary:
            if not sequential_parfor_lowering:
                print_wrapped(' Fusing loops '.center(_termwidth, '-'))
                msg = 'Attempting fusion of parallel loops (combines loops with similar properties)...\n'
                print_wrapped(msg)
            else:
                msg = 'Performing sequential lowering of loops...\n'
                print_wrapped(msg)
                print_wrapped(_termwidth * '-')
        if n_parfors > -1:

            def dump_graph_indented(a, root_msg, node_msg):
                fac = len(sword)

                def print_graph(adj, roots):

                    def print_g(adj, root, depth):
                        for k in adj[root]:
                            print_wrapped(fac * depth * ' ' + '%s%s %s' % (sword, k, node_msg))
                            if adj[k] != []:
                                print_g(adj, k, depth + 1)
                    for r in roots:
                        print_wrapped('%s%s %s' % (sword, r, root_msg))
                        print_g(l, r, 1)
                        print_wrapped('')
                l, roots = self.compute_graph_info(a)
                print_graph(l, roots)
            if print_fusion_search:
                for report in self.fusion_reports:
                    l1, l2, msg = report
                    print_wrapped('  Trying to fuse loops #%s and #%s:' % (l1, l2))
                    print_wrapped('    %s' % msg)
            if self.fusion_info != {}:
                if print_fusion_summary:
                    print_wrapped('\n \nFused loop summary:\n')
                    dump_graph_indented(self.fusion_info, 'has the following loops fused into it:', '(fused)')
            if print_fusion_summary:
                if self.fusion_enabled:
                    after_fusion = 'Following the attempted fusion of parallel for-loops'
                else:
                    after_fusion = 'With fusion disabled'
                print_wrapped('\n{} there are {} parallel for-loop(s) (originating from loops labelled: {}).'.format(after_fusion, n_parfors, ', '.join(['#%s' % x for x in parfor_ids])))
                print_wrapped(_termwidth * '-')
                print_wrapped('')
            if print_loopnest_rewrite:
                if self.nested_fusion_info != {}:
                    print_wrapped(' Optimising loop nests '.center(_termwidth, '-'))
                    print_wrapped('Attempting loop nest rewrites (optimising for the largest parallel loops)...\n ')
                    root_msg = 'is a parallel loop'
                    node_msg = '--> rewritten as a serial loop'
                    dump_graph_indented(self.nested_fusion_info, root_msg, node_msg)
                    print_wrapped(_termwidth * '-')
                    print_wrapped('')
            all_lines = self.get_all_lines(parfors_simple)
            if print_pre_optimised:
                print(' Before Optimisation '.center(_termwidth, '-'))
                self.print_unoptimised(all_lines)
                print(_termwidth * '-')
            if print_post_optimised:
                print(' After Optimisation '.center(_termwidth, '-'))
                self.print_optimised(all_lines)
                print(_termwidth * '-')
            print_wrapped('')
            print_wrapped(_termwidth * '-')
            print_wrapped('\n ')
            if print_allocation_hoist or print_instruction_hoist:
                print_wrapped('Loop invariant code motion'.center(80, '-'))
            if print_allocation_hoist:
                self.allocation_hoist()
            if print_instruction_hoist:
                self.instruction_hoist()
        else:
            print_wrapped('Function %s, %s, has no parallel for-loops.'.format(name, line))

    def __str__(self):
        r = 'ParforDiagnostics:\n'
        r += repr(self.replaced_fns)
        return r

    def __repr__(self):
        r = 'ParforDiagnostics'
        return r
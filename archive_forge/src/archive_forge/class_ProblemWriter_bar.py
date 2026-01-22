import itertools
import logging
import math
from io import StringIO
from contextlib import nullcontext
from pyomo.common.collections import OrderedSet
from pyomo.opt import ProblemFormat
from pyomo.opt.base import AbstractProblemWriter, WriterFactory
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.visitor import _ToStringVisitor
import pyomo.core.expr as EXPR
from pyomo.core.base import (
from pyomo.core.base.component import ActiveComponent
import pyomo.core.base.suffix
import pyomo.core.kernel.suffix
from pyomo.core.kernel.block import IBlock
from pyomo.repn.util import valid_expr_ctypes_minlp, valid_active_ctypes_minlp, ftoa
@WriterFactory.register('bar', 'Generate the corresponding BARON BAR file.')
class ProblemWriter_bar(AbstractProblemWriter):

    def __init__(self):
        AbstractProblemWriter.__init__(self, ProblemFormat.bar)

    def _write_equations_section(self, model, output_file, all_blocks_list, active_components_data_var, symbol_map, c_labeler, output_fixed_variable_bounds, skip_trivial_constraints, sorter):
        referenced_variable_ids = OrderedSet()

        def _skip_trivial(constraint_data):
            if skip_trivial_constraints:
                if constraint_data._linear_canonical_form:
                    repn = constraint_data.canonical_form()
                    if repn.variables is None or len(repn.variables) == 0:
                        return True
                elif constraint_data.body.polynomial_degree() == 0:
                    return True
            return False
        if isinstance(model, IBlock):
            suffix_gen = lambda b: ((suf.storage_key, suf) for suf in pyomo.core.kernel.suffix.export_suffix_generator(b, active=True, descend_into=False))
        else:
            suffix_gen = lambda b: pyomo.core.base.suffix.active_export_suffix_generator(b)
        r_o_eqns = []
        c_eqns = []
        l_eqns = []
        branching_priorities_suffixes = []
        for block in all_blocks_list:
            for name, suffix in suffix_gen(block):
                if name in {'branching_priorities', 'priority'}:
                    branching_priorities_suffixes.append(suffix)
                elif name == 'constraint_types':
                    for constraint_data, constraint_type in suffix.items():
                        if not _skip_trivial(constraint_data):
                            if constraint_type.lower() == 'relaxationonly':
                                r_o_eqns.append(constraint_data)
                            elif constraint_type.lower() == 'convex':
                                c_eqns.append(constraint_data)
                            elif constraint_type.lower() == 'local':
                                l_eqns.append(constraint_data)
                            else:
                                raise ValueError("A suffix '%s' contained an invalid value: %s\nChoices are: [relaxationonly, convex, local]" % (suffix.name, constraint_type))
                else:
                    if block is block.model():
                        if block.name == 'unknown':
                            _location = 'model'
                        else:
                            _location = "model '%s'" % (block.name,)
                    else:
                        _location = "block '%s'" % (block.name,)
                    raise ValueError("The BARON writer can not export suffix with name '%s'. Either remove it from the %s or deactivate it." % (name, _location))
        non_standard_eqns = r_o_eqns + c_eqns + l_eqns
        n_roeqns = len(r_o_eqns)
        n_ceqns = len(c_eqns)
        n_leqns = len(l_eqns)
        eqns = []
        order_counter = 0
        alias_template = '.c%d'
        output_file.write('EQUATIONS ')
        output_file.write('c_e_FIX_ONE_VAR_CONST__')
        order_counter += 1
        for block in all_blocks_list:
            for constraint_data in block.component_data_objects(Constraint, active=True, sort=sorter, descend_into=False):
                if not constraint_data.has_lb() and (not constraint_data.has_ub()):
                    assert not constraint_data.equality
                    continue
                if not _skip_trivial(constraint_data) and constraint_data not in non_standard_eqns:
                    eqns.append(constraint_data)
                    con_symbol = symbol_map.createSymbol(constraint_data, c_labeler)
                    assert not con_symbol.startswith('.')
                    assert con_symbol != 'c_e_FIX_ONE_VAR_CONST__'
                    symbol_map.alias(constraint_data, alias_template % order_counter)
                    output_file.write(', ' + str(con_symbol))
                    order_counter += 1
        output_file.write(';\n\n')
        if n_roeqns > 0:
            output_file.write('RELAXATION_ONLY_EQUATIONS ')
            for i, constraint_data in enumerate(r_o_eqns):
                con_symbol = symbol_map.createSymbol(constraint_data, c_labeler)
                assert not con_symbol.startswith('.')
                assert con_symbol != 'c_e_FIX_ONE_VAR_CONST__'
                symbol_map.alias(constraint_data, alias_template % order_counter)
                if i == n_roeqns - 1:
                    output_file.write(str(con_symbol) + ';\n\n')
                else:
                    output_file.write(str(con_symbol) + ', ')
                order_counter += 1
        if n_ceqns > 0:
            output_file.write('CONVEX_EQUATIONS ')
            for i, constraint_data in enumerate(c_eqns):
                con_symbol = symbol_map.createSymbol(constraint_data, c_labeler)
                assert not con_symbol.startswith('.')
                assert con_symbol != 'c_e_FIX_ONE_VAR_CONST__'
                symbol_map.alias(constraint_data, alias_template % order_counter)
                if i == n_ceqns - 1:
                    output_file.write(str(con_symbol) + ';\n\n')
                else:
                    output_file.write(str(con_symbol) + ', ')
                order_counter += 1
        if n_leqns > 0:
            output_file.write('LOCAL_EQUATIONS ')
            for i, constraint_data in enumerate(l_eqns):
                con_symbol = symbol_map.createSymbol(constraint_data, c_labeler)
                assert not con_symbol.startswith('.')
                assert con_symbol != 'c_e_FIX_ONE_VAR_CONST__'
                symbol_map.alias(constraint_data, alias_template % order_counter)
                if i == n_leqns - 1:
                    output_file.write(str(con_symbol) + ';\n\n')
                else:
                    output_file.write(str(con_symbol) + ', ')
                order_counter += 1
        if isinstance(model, IBlock):
            mutable_param_gen = lambda b: b.components(ctype=Param, descend_into=False)
        else:

            def mutable_param_gen(b):
                for param in block.component_objects(Param):
                    if param.mutable and param.is_indexed():
                        param_data_iter = (param_data for index, param_data in param.items())
                    elif not param.is_indexed():
                        param_data_iter = iter([param])
                    else:
                        param_data_iter = iter([])
                    for param_data in param_data_iter:
                        yield param_data
        output_file.write('c_e_FIX_ONE_VAR_CONST__:  ONE_VAR_CONST__  == 1;\n')
        for constraint_data in itertools.chain(eqns, r_o_eqns, c_eqns, l_eqns):
            variables = OrderedSet()
            eqn_body = expression_to_string(constraint_data.body, variables, smap=symbol_map)
            referenced_variable_ids.update(variables)
            if len(variables) == 0:
                assert not skip_trivial_constraints
                eqn_body += ' + 0 * ONE_VAR_CONST__ '
            con_symbol = symbol_map.byObject[id(constraint_data)]
            output_file.write(str(con_symbol) + ': ')
            if constraint_data.equality:
                eqn_lhs = ''
                eqn_rhs = ' == ' + ftoa(constraint_data.upper)
            elif not constraint_data.has_ub():
                eqn_rhs = ' >= ' + ftoa(constraint_data.lower)
                eqn_lhs = ''
            elif not constraint_data.has_lb():
                eqn_rhs = ' <= ' + ftoa(constraint_data.upper)
                eqn_lhs = ''
            elif constraint_data.has_lb() and constraint_data.has_ub():
                eqn_lhs = ftoa(constraint_data.lower) + ' <= '
                eqn_rhs = ' <= ' + ftoa(constraint_data.upper)
            eqn_string = eqn_lhs + eqn_body + eqn_rhs + ';\n'
            output_file.write(eqn_string)
        output_file.write('\nOBJ: ')
        n_objs = 0
        for block in all_blocks_list:
            for objective_data in block.component_data_objects(Objective, active=True, sort=sorter, descend_into=False):
                n_objs += 1
                if n_objs > 1:
                    raise ValueError('The BARON writer has detected multiple active objective functions on model %s, but currently only handles a single objective.' % model.name)
                symbol_map.createSymbol(objective_data, c_labeler)
                symbol_map.alias(objective_data, '__default_objective__')
                if objective_data.is_minimizing():
                    output_file.write('minimize ')
                else:
                    output_file.write('maximize ')
                variables = OrderedSet()
                obj_string = expression_to_string(objective_data.expr, variables, smap=symbol_map)
                referenced_variable_ids.update(variables)
        output_file.write(obj_string + ';\n\n')
        return (referenced_variable_ids, branching_priorities_suffixes)

    def __call__(self, model, output_filename, solver_capability, io_options):
        if output_filename is None:
            output_filename = model.name + '.bar'
        if isinstance(output_filename, str):
            output_file = open(output_filename, 'w')
        else:
            output_file = nullcontext(output_filename)
        with output_file as FILE:
            symbol_map = self._write_bar_file(model, FILE, solver_capability, io_options)
        return (output_filename, symbol_map)

    def _write_bar_file(self, model, output_file, solver_capability, io_options):
        io_options = dict(io_options)
        symbolic_solver_labels = io_options.pop('symbolic_solver_labels', False)
        labeler = io_options.pop('labeler', None)
        file_determinism = io_options.pop('file_determinism', 1)
        sorter = SortComponents.unsorted
        if file_determinism >= 1:
            sorter = sorter | SortComponents.indices
            if file_determinism >= 2:
                sorter = sorter | SortComponents.alphabetical
        output_fixed_variable_bounds = io_options.pop('output_fixed_variable_bounds', False)
        skip_trivial_constraints = io_options.pop('skip_trivial_constraints', False)
        solver_options = io_options.pop('solver_options', {})
        if len(io_options):
            raise ValueError('ProblemWriter_baron_writer passed unrecognized io_options:\n\t' + '\n\t'.join(('%s = %s' % (k, v) for k, v in io_options.items())))
        if symbolic_solver_labels and labeler is not None:
            raise ValueError("Baron problem writer: Using both the 'symbolic_solver_labels' and 'labeler' I/O options is forbidden")
        model_ctypes = model.collect_ctypes(active=True)
        invalids = set()
        for t in model_ctypes - valid_active_ctypes_minlp:
            if issubclass(t, ActiveComponent):
                invalids.add(t)
        if len(invalids):
            invalids = [t.__name__ for t in invalids]
            raise RuntimeError('Unallowable active component(s) %s.\nThe BARON writer cannot export models with this component type.' % ', '.join(invalids))
        output_file.write('OPTIONS {\n')
        summary_found = False
        if len(solver_options):
            for key, val in solver_options.items():
                if key.lower() == 'summary':
                    summary_found = True
                if key.endswith('Name'):
                    output_file.write(key + ': "' + str(val) + '";\n')
                else:
                    output_file.write(key + ': ' + str(val) + ';\n')
        if not summary_found:
            output_file.write('Summary: 0;\n')
        output_file.write('}\n\n')
        if symbolic_solver_labels:
            v_labeler = c_labeler = ShortNameLabeler(15, prefix='s_', suffix='_', caseInsensitive=True, legalRegex='^[a-zA-Z]')
        elif labeler is None:
            v_labeler = NumericLabeler('x')
            c_labeler = NumericLabeler('c')
        else:
            v_labeler = c_labeler = labeler
        symbol_map = SymbolMap()
        symbol_map.default_labeler = v_labeler
        all_blocks_list = list(model.block_data_objects(active=True, sort=sorter, descend_into=True))
        active_components_data_var = {}
        equation_section_stream = StringIO()
        referenced_variable_ids, branching_priorities_suffixes = self._write_equations_section(model, equation_section_stream, all_blocks_list, active_components_data_var, symbol_map, c_labeler, output_fixed_variable_bounds, skip_trivial_constraints, sorter)
        BinVars = []
        IntVars = []
        PosVars = []
        Vars = []
        for vid in referenced_variable_ids:
            name = symbol_map.byObject[vid]
            var_data = symbol_map.bySymbol[name]
            if var_data.is_continuous():
                if var_data.has_lb() and value(var_data.lb) >= 0:
                    TypeList = PosVars
                else:
                    TypeList = Vars
            elif var_data.is_binary():
                TypeList = BinVars
            elif var_data.is_integer():
                TypeList = IntVars
            else:
                assert False
            TypeList.append(name)
        if len(BinVars) > 0:
            BinVars.sort()
            output_file.write('BINARY_VARIABLES ')
            output_file.write(', '.join(BinVars))
            output_file.write(';\n\n')
        if len(IntVars) > 0:
            IntVars.sort()
            output_file.write('INTEGER_VARIABLES ')
            output_file.write(', '.join(IntVars))
            output_file.write(';\n\n')
        PosVars.append('ONE_VAR_CONST__')
        PosVars.sort()
        output_file.write('POSITIVE_VARIABLES ')
        output_file.write(', '.join(PosVars))
        output_file.write(';\n\n')
        if len(Vars) > 0:
            Vars.sort()
            output_file.write('VARIABLES ')
            output_file.write(', '.join(Vars))
            output_file.write(';\n\n')
        lbounds = {}
        for vid in referenced_variable_ids:
            name = symbol_map.byObject[vid]
            var_data = symbol_map.bySymbol[name]
            if var_data.fixed:
                if output_fixed_variable_bounds:
                    var_data_lb = ftoa(var_data.value, False)
                else:
                    var_data_lb = None
            else:
                var_data_lb = None
                if var_data.has_lb():
                    var_data_lb = ftoa(var_data.lb, False)
            if var_data_lb is not None:
                name_to_output = symbol_map.getSymbol(var_data)
                lbounds[name_to_output] = '%s: %s;\n' % (name_to_output, var_data_lb)
        if len(lbounds) > 0:
            output_file.write('LOWER_BOUNDS{\n')
            output_file.write(''.join((lbounds[key] for key in sorted(lbounds.keys()))))
            output_file.write('}\n\n')
        lbounds = None
        ubounds = {}
        for vid in referenced_variable_ids:
            name = symbol_map.byObject[vid]
            var_data = symbol_map.bySymbol[name]
            if var_data.fixed:
                if output_fixed_variable_bounds:
                    var_data_ub = ftoa(var_data.value, False)
                else:
                    var_data_ub = None
            else:
                var_data_ub = None
                if var_data.has_ub():
                    var_data_ub = ftoa(var_data.ub, False)
            if var_data_ub is not None:
                name_to_output = symbol_map.getSymbol(var_data)
                ubounds[name_to_output] = '%s: %s;\n' % (name_to_output, var_data_ub)
        if len(ubounds) > 0:
            output_file.write('UPPER_BOUNDS{\n')
            output_file.write(''.join((ubounds[key] for key in sorted(ubounds.keys()))))
            output_file.write('}\n\n')
        ubounds = None
        BranchingPriorityHeader = False
        for suffix in branching_priorities_suffixes:
            for var, priority in suffix.items():
                if var.is_indexed():
                    var_iter = var.values()
                else:
                    var_iter = (var,)
                for var_data in var_iter:
                    if id(var_data) not in referenced_variable_ids:
                        continue
                    if priority is not None:
                        if not BranchingPriorityHeader:
                            output_file.write('BRANCHING_PRIORITIES{\n')
                            BranchingPriorityHeader = True
                        output_file.write('%s: %s;\n' % (symbol_map.getSymbol(var_data), priority))
        if BranchingPriorityHeader:
            output_file.write('}\n\n')
        output_file.write(equation_section_stream.getvalue())
        output_file.write('STARTING_POINT{\nONE_VAR_CONST__: 1;\n')
        tmp = {}
        for vid in referenced_variable_ids:
            name = symbol_map.byObject[vid]
            var_data = symbol_map.bySymbol[name]
            starting_point = var_data.value
            if starting_point is not None:
                var_name = symbol_map.getSymbol(var_data)
                tmp[var_name] = '%s: %s;\n' % (var_name, ftoa(starting_point, False))
        output_file.write(''.join((tmp[key] for key in sorted(tmp.keys()))))
        output_file.write('}\n\n')
        return symbol_map
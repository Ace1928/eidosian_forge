import re
from io import BytesIO
from .lazy_import import lazy_import
from fnmatch import fnmatch
from breezy._termcolor import color_string, FG
from breezy import (
from . import controldir, errors, osutils
from . import revision as _mod_revision
from . import trace
from .revisionspec import RevisionSpec, RevisionSpec_revid, RevisionSpec_revno
def versioned_grep(opts):
    wt, branch, relpath = controldir.ControlDir.open_containing_tree_or_branch('.')
    with branch.lock_read():
        start_rev = opts.revision[0]
        start_revid = start_rev.as_revision_id(branch)
        if start_revid is None:
            start_rev = RevisionSpec_revno.from_string('revno:1')
            start_revid = start_rev.as_revision_id(branch)
        srevno_tuple = branch.revision_id_to_dotted_revno(start_revid)
        if len(opts.revision) == 2:
            end_rev = opts.revision[1]
            end_revid = end_rev.as_revision_id(branch)
            if end_revid is None:
                end_revno, end_revid = branch.last_revision_info()
            erevno_tuple = branch.revision_id_to_dotted_revno(end_revid)
            grep_mainline = _rev_on_mainline(srevno_tuple) and _rev_on_mainline(erevno_tuple)
            if srevno_tuple > erevno_tuple:
                srevno_tuple, erevno_tuple = (erevno_tuple, srevno_tuple)
                start_revid, end_revid = (end_revid, start_revid)
            if opts.levels == 1 and grep_mainline:
                given_revs = _linear_view_revisions(branch, start_revid, end_revid)
            else:
                given_revs = _graph_view_revisions(branch, start_revid, end_revid)
        else:
            start_revno = '.'.join(map(str, srevno_tuple))
            start_rev_tuple = (start_revid, start_revno, 0)
            given_revs = [start_rev_tuple]
        opts.outputter = _Outputter(opts, use_cache=True)
        for revid, revno, merge_depth in given_revs:
            if opts.levels == 1 and merge_depth != 0:
                continue
            rev = RevisionSpec_revid.from_string('revid:' + revid.decode('utf-8'))
            tree = rev.as_tree(branch)
            for path in opts.path_list:
                tree_path = osutils.pathjoin(relpath, path)
                if not tree.has_filename(tree_path):
                    trace.warning("Skipped unknown file '%s'.", path)
                    continue
                if osutils.isdir(path):
                    path_prefix = path
                    dir_grep(tree, path, relpath, opts, revno, path_prefix)
                else:
                    versioned_file_grep(tree, tree_path, '.', path, opts, revno)
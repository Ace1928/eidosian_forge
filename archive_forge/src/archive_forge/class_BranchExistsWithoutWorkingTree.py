class BranchExistsWithoutWorkingTree(PathError):
    _fmt = 'Directory contains a branch, but no working tree (use brz checkout if you wish to build a working tree): "%(path)s"'
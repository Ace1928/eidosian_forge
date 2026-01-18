import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
def try_dispatch(self, sources, src_dir=None, ccompiler=None, **kwargs):
    """
        Compile one or more dispatch-able sources and generates object files,
        also generates abstract C config headers and macros that
        used later for the final runtime dispatching process.

        The mechanism behind it is to takes each source file that specified
        in 'sources' and branching it into several files depend on
        special configuration statements that must be declared in the
        top of each source which contains targeted CPU features,
        then it compiles every branched source with the proper compiler flags.

        Parameters
        ----------
        sources : list
            Must be a list of dispatch-able sources file paths,
            and configuration statements must be declared inside
            each file.

        src_dir : str
            Path of parent directory for the generated headers and wrapped sources.
            If None(default) the files will generated in-place.

        ccompiler : CCompiler
            Distutils `CCompiler` instance to be used for compilation.
            If None (default), the provided instance during the initialization
            will be used instead.

        **kwargs : any
            Arguments to pass on to the `CCompiler.compile()`

        Returns
        -------
        list : generated object files

        Raises
        ------
        CompileError
            Raises by `CCompiler.compile()` on compiling failure.
        DistutilsError
            Some errors during checking the sanity of configuration statements.

        See Also
        --------
        parse_targets :
            Parsing the configuration statements of dispatch-able sources.
        """
    to_compile = {}
    baseline_flags = self.cpu_baseline_flags()
    include_dirs = kwargs.setdefault('include_dirs', [])
    for src in sources:
        output_dir = os.path.dirname(src)
        if src_dir:
            if not output_dir.startswith(src_dir):
                output_dir = os.path.join(src_dir, output_dir)
            if output_dir not in include_dirs:
                include_dirs.append(output_dir)
        has_baseline, targets, extra_flags = self.parse_targets(src)
        nochange = self._generate_config(output_dir, src, targets, has_baseline)
        for tar in targets:
            tar_src = self._wrap_target(output_dir, src, tar, nochange=nochange)
            flags = tuple(extra_flags + self.feature_flags(tar))
            to_compile.setdefault(flags, []).append(tar_src)
        if has_baseline:
            flags = tuple(extra_flags + baseline_flags)
            to_compile.setdefault(flags, []).append(src)
        self.sources_status[src] = (has_baseline, targets)
    objects = []
    for flags, srcs in to_compile.items():
        objects += self.dist_compile(srcs, list(flags), ccompiler=ccompiler, **kwargs)
    return objects
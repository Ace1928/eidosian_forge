from ... import controldir
from ...commands import Command
from ...option import Option, RegistryOption
from . import helpers, load_fastimport
class cmd_fast_export(Command):
    """Generate a fast-import stream from a Bazaar branch.

    This program generates a stream from a Bazaar branch in fast-import
    format used by tools such as bzr fast-import, git-fast-import and
    hg-fast-import.

    It takes two optional arguments: the source bzr branch to export and
    the destination to write the file to write the fastimport stream to.

    If no source is specified, it will search for a branch in the
    current directory.

    If no destination is given or the destination is '-', standard output
    is used. Otherwise, the destination is the name of a file. If the
    destination ends in '.gz', the output will be compressed into gzip
    format.

    :Round-tripping:

     Recent versions of the fast-import specification support features
     that allow effective round-tripping most of the metadata in Bazaar
     branches. As such, fast-exporting a branch and fast-importing the data
     produced will create a new repository with roughly equivalent history, i.e.
     "bzr log -v -p --include-merges --forward" on the old branch and
     new branch should produce similar, if not identical, results.

     .. note::

        Be aware that the new repository may appear to have similar history
        but internally it is quite different with new revision-ids and
        file-ids assigned. As a consequence, the ability to easily merge
        with branches based on the old repository is lost. Depending on your
        reasons for producing a new repository, this may or may not be an
        issue.

    :Interoperability:

     fast-export can use the following "extended features" to
     produce a richer data stream:

     * *multiple-authors* - if a commit has multiple authors (as commonly
       occurs in pair-programming), all authors will be included in the
       output, not just the first author

     * *commit-properties* - custom metadata per commit that Bazaar stores
       in revision properties (e.g. branch-nick and bugs fixed by this
       change) will be included in the output.

     * *empty-directories* - directories, even the empty ones, will be
       included in the output.

     To disable these features and produce output acceptable to git 1.6,
     use the --plain option. To enable these features, use --no-plain.
     Currently, --plain is the default but that will change in the near
     future once the feature names and definitions are formally agreed
     to by the broader fast-import developer community.

     Git has stricter naming rules for tags and fast-export --plain
     will skip tags which can't be imported into git. To replace characters
     unsupported in git with an underscore instead, specify
     --rewrite-tag-names.

    :History truncation:

     It is sometimes convenient to simply truncate the revision history at a
     certain point.  The --baseline option, to be used in conjunction with -r,
     emits a baseline commit containing the state of the entire source tree at
     the first requested revision.  This allows a user to produce a tree
     identical to the original without munging multiple exports.

    :Examples:

     To produce data destined for import into Bazaar::

       bzr fast-export --no-plain my-bzr-branch my.fi.gz

     To produce data destined for Git 1.6::

       bzr fast-export --plain my-bzr-branch my.fi

     To import several unmerged but related branches into the same repository,
     use the --{export,import}-marks options, and specify a name for the git
     branch like this::

       bzr fast-export --export-marks=marks.bzr project.dev |
              GIT_DIR=project/.git git-fast-import --export-marks=marks.git

       bzr fast-export --import-marks=marks.bzr -b other project.other |
              GIT_DIR=project/.git git-fast-import --import-marks=marks.git

     If you get a "Missing space after source" error from git-fast-import,
     see the top of the commands.py module for a work-around.

     Since bzr uses per-branch tags and git/hg use per-repo tags, the
     way bzr fast-export presently emits tags (unconditional reset &
     new ref) may result in clashes when several different branches
     are imported into single git/hg repo.  If this occurs, use the
     bzr fast-export option --no-tags during the export of one or more
     branches to avoid the issue.
    """
    hidden = False
    _see_also = ['fast-import', 'fast-import-filter']
    takes_args = ['source?', 'destination?']
    takes_options = ['verbose', 'revision', Option('git-branch', short_name='b', type=str, argname='FILE', help='Name of the git branch to create (default=master).'), Option('checkpoint', type=int, argname='N', help='Checkpoint every N revisions (default=10000).'), Option('marks', type=str, argname='FILE', help='Import marks from and export marks to file.'), Option('import-marks', type=str, argname='FILE', help='Import marks from file.'), Option('export-marks', type=str, argname='FILE', help='Export marks to file.'), Option('plain', help='Exclude metadata to maximise interoperability.'), Option('rewrite-tag-names', help="Replace characters invalid in git with '_' (plain mode only)."), Option('baseline', help="Export an 'absolute' baseline commit prior tothe first relative commit"), Option('no-tags', help="Don't export tags")]
    encoding_type = 'exact'

    def run(self, source=None, destination=None, verbose=False, git_branch='master', checkpoint=10000, marks=None, import_marks=None, export_marks=None, revision=None, plain=True, rewrite_tag_names=False, no_tags=False, baseline=False):
        load_fastimport()
        from ...branch import Branch
        from . import exporter
        if marks:
            import_marks = export_marks = marks
        if source is None:
            source = '.'
        branch = Branch.open_containing(source)[0]
        outf = exporter._get_output_stream(destination)
        exporter = exporter.BzrFastExporter(branch, outf=outf, ref=b'refs/heads/%s' % git_branch.encode('utf-8'), checkpoint=checkpoint, import_marks_file=import_marks, export_marks_file=export_marks, revision=revision, verbose=verbose, plain_format=plain, rewrite_tags=rewrite_tag_names, no_tags=no_tags, baseline=baseline)
        return exporter.run()
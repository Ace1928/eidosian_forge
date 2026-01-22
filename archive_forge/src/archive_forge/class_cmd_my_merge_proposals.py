from io import StringIO
from ... import branch as _mod_branch
from ... import controldir, errors
from ... import forge as _mod_forge
from ... import log as _mod_log
from ... import missing as _mod_missing
from ... import msgeditor, urlutils
from ...commands import Command
from ...i18n import gettext
from ...option import ListOption, Option, RegistryOption
from ...trace import note, warning
class cmd_my_merge_proposals(Command):
    __doc__ = 'List all merge proposals owned by the logged-in user.\n\n    '
    hidden = True
    takes_args = ['base_url?']
    takes_options = ['verbose', RegistryOption.from_kwargs('status', title='Proposal Status', help='Only include proposals with specified status.', value_switches=True, enum_switch=True, all='All merge proposals', open='Open merge proposals', merged='Merged merge proposals', closed='Closed merge proposals'), RegistryOption('forge', help='Use the forge.', lazy_registry=('breezy.forge', 'forges'))]

    def run(self, status='open', verbose=False, forge=None, base_url=None):
        for instance in _mod_forge.iter_forge_instances(forge=forge):
            if base_url is not None and instance.base_url != base_url:
                continue
            try:
                for mp in instance.iter_my_proposals(status=status):
                    self.outf.write('%s\n' % mp.url)
                    if verbose:
                        source_branch_url = mp.get_source_branch_url()
                        if source_branch_url:
                            self.outf.write('(Merging %s into %s)\n' % (source_branch_url, mp.get_target_branch_url()))
                        else:
                            self.outf.write('(Merging into %s)\n' % mp.get_target_branch_url())
                        description = mp.get_description()
                        if description:
                            self.outf.writelines(['\t%s\n' % l for l in description.splitlines()])
                        self.outf.write('\n')
            except _mod_forge.ForgeLoginRequired as e:
                warning('Skipping %s, login required.', instance)
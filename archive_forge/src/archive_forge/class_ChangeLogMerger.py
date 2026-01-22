import difflib
import patiencediff
from merge3 import Merge3
from ... import debug, merge, osutils
from ...trace import mutter
class ChangeLogMerger(merge.ConfigurableFileMerger):
    """Merge GNU-format ChangeLog files."""
    name_prefix = 'changelog'

    def file_matches(self, params):
        affected_files = self.affected_files
        if affected_files is None:
            config = self.merger.this_branch.get_config()
            config_key = self.name_prefix + '_merge_files'
            affected_files = config.get_user_option_as_list(config_key)
            if affected_files is None:
                affected_files = self.default_files
            self.affected_files = affected_files
        if affected_files:
            filepath = osutils.basename(params.this_path)
            if filepath in affected_files:
                return True
        return False

    def merge_text(self, params):
        """Merge changelog changes.

         * new entries from other will float to the top
         * edits to older entries are preserved
        """
        this_entries = changelog_entries(params.this_lines)
        other_entries = changelog_entries(params.other_lines)
        base_entries = changelog_entries(params.base_lines)
        try:
            result_entries = merge_entries(base_entries, this_entries, other_entries)
        except EntryConflict:
            return ('not_applicable', None)
        return ('success', entries_to_lines(result_entries))
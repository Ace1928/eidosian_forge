import sys, os
import textwrap
def _get_all_options(self):
    options = self.option_list[:]
    for group in self.option_groups:
        options.extend(group.option_list)
    return options
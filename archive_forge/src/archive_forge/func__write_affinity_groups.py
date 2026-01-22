from __future__ import (absolute_import, division, print_function)
import sys
import getopt
import logging
import ovirtsdk4 as sdk
import ovirtsdk4.types as otypes
def _write_affinity_groups(f, affinity_groups):
    f.write('\n# Mapping for affinity group\n')
    f.write('dr_affinity_group_mappings:\n')
    for affinity_group in affinity_groups:
        f.write('- primary_name: %s\n' % affinity_group)
        f.write("  # Fill the correlated affinity group name in the secondary site for affinity '%s'\n" % affinity_group)
        f.write('  secondary_name: # %s\n\n' % affinity_group)
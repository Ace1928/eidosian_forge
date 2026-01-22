from __future__ import (absolute_import, division, print_function)
import sys
import getopt
import logging
import ovirtsdk4 as sdk
import ovirtsdk4.types as otypes
def _write_aaa_domains(f, domains):
    f.write('\n# Mapping for domain\n')
    f.write('dr_domain_mappings: \n')
    for domain in domains:
        f.write('- primary_name: %s\n' % domain)
        f.write("  # Fill in the correlated domain in the secondary site for domain '%s'\n" % domain)
        f.write('  secondary_name: # %s\n\n' % domain)
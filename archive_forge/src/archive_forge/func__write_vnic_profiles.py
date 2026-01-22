from __future__ import (absolute_import, division, print_function)
import sys
import getopt
import logging
import ovirtsdk4 as sdk
import ovirtsdk4.types as otypes
def _write_vnic_profiles(f, networks):
    f.write('dr_network_mappings:\n')
    for network in networks:
        f.write('- primary_network_name: %s\n' % network['network_name'])
        f.write('# Data Center name is relevant when multiple vnic profiles are maintained.\n')
        f.write('# please uncomment it in case you have more than one DC.\n')
        f.write('# primary_network_dc: %s\n' % network['network_dc'])
        f.write('  primary_profile_name: %s\n' % network['profile_name'])
        f.write('  primary_profile_id: %s\n' % network['profile_id'])
        f.write("  # Fill in the correlated vnic profile properties in the secondary site for profile '%s'\n" % network['profile_name'])
        f.write('  secondary_network_name: # %s\n' % network['network_name'])
        f.write('# Data Center name is relevant when multiple vnic profiles are maintained.\n')
        f.write('# please uncomment it in case you have more than one DC.\n')
        f.write('# secondary_network_dc: %s\n' % network['network_dc'])
        f.write('  secondary_profile_name: # %s\n' % network['profile_name'])
        f.write('  secondary_profile_id: # %s\n\n' % network['profile_id'])
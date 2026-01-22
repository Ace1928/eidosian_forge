from __future__ import (absolute_import, division, print_function)
import sys
import getopt
import logging
import ovirtsdk4 as sdk
import ovirtsdk4.types as otypes
def _get_aaa_domains(connection):
    domains = []
    domains_service = connection.system_service().domains_service()
    domains_list = domains_service.list()
    for domain in domains_list:
        domains.append(domain.name)
    return domains
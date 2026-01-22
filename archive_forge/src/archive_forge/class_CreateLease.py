import argparse
import datetime
import logging
import re
from oslo_serialization import jsonutils
from oslo_utils import strutils
from blazarclient import command
from blazarclient import exception
class CreateLease(CreateLeaseBase):

    def get_parser(self, prog_name):
        parser = super(CreateLease, self).get_parser(prog_name)
        parser.add_argument('--physical-reservation', metavar='<min=int,max=int,hypervisor_properties=str,resource_properties=str,before_end=str>', action='append', dest='physical_reservations', help='Create a reservation for physical compute hosts. Specify option multiple times to create multiple reservations. min: minimum number of hosts to reserve. max: maximum number of hosts to reserve. hypervisor_properties: JSON string, see doc. resource_properties: JSON string, see doc. before_end: JSON string, see doc. ', default=[])
        return parser

    def args2body(self, parsed_args):
        params = self._generate_params(parsed_args)
        physical_reservations = []
        for phys_res_str in parsed_args.physical_reservations:
            err_msg = "Invalid physical-reservation argument '%s'. Reservation arguments must be of the form --physical-reservation <min=int,max=int,hypervisor_properties=str,resource_properties=str,before_end=str>" % phys_res_str
            defaults = CREATE_RESERVATION_KEYS['physical:host']
            phys_res_info = self._parse_params(phys_res_str, defaults, err_msg)
            if not (phys_res_info['min'] and phys_res_info['max']):
                raise exception.IncorrectLease(err_msg)
            if not (strutils.is_int_like(phys_res_info['min']) and strutils.is_int_like(phys_res_info['max'])):
                raise exception.IncorrectLease(err_msg)
            min_host = int(phys_res_info['min'])
            max_host = int(phys_res_info['max'])
            if min_host > max_host:
                err_msg = "Invalid physical-reservation argument '%s'. Reservation argument min value must be less than max value" % phys_res_str
                raise exception.IncorrectLease(err_msg)
            if min_host == 0 or max_host == 0:
                err_msg = "Invalid physical-reservation argument '%s'. Reservation arguments min and max values must be greater than or equal to 1" % phys_res_str
                raise exception.IncorrectLease(err_msg)
            phys_res_info['resource_type'] = 'physical:host'
            physical_reservations.append(phys_res_info)
        if physical_reservations:
            params['reservations'] = physical_reservations + params['reservations']
        return params
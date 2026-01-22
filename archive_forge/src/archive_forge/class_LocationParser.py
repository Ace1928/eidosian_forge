import os
from collections import namedtuple
from ..common.exceptions import DWARFError
from ..common.utils import struct_parse
from .dwarf_util import _iter_CUs_in_section
class LocationParser(object):
    """ A parser for location information in DIEs.
        Handles both location information contained within the attribute
        itself (represented as a LocationExpr object) and references to
        location lists in the .debug_loc section (represented as a
        list).
    """

    def __init__(self, location_lists):
        self.location_lists = location_lists

    @staticmethod
    def attribute_has_location(attr, dwarf_version):
        """ Checks if a DIE attribute contains location information.
        """
        return LocationParser._attribute_is_loclistptr_class(attr) and (LocationParser._attribute_has_loc_expr(attr, dwarf_version) or LocationParser._attribute_has_loc_list(attr, dwarf_version))

    def parse_from_attribute(self, attr, dwarf_version, die=None):
        """ Parses a DIE attribute and returns either a LocationExpr or
            a list.
        """
        if self.attribute_has_location(attr, dwarf_version):
            if self._attribute_has_loc_expr(attr, dwarf_version):
                return LocationExpr(attr.value)
            elif self._attribute_has_loc_list(attr, dwarf_version):
                return self.location_lists.get_location_list_at_offset(attr.value, die)
        else:
            raise ValueError('Attribute does not have location information')

    @staticmethod
    def _attribute_has_loc_expr(attr, dwarf_version):
        return dwarf_version < 4 and attr.form.startswith('DW_FORM_block') and (not attr.name == 'DW_AT_const_value') or attr.form == 'DW_FORM_exprloc'

    @staticmethod
    def _attribute_has_loc_list(attr, dwarf_version):
        return (dwarf_version < 4 and attr.form in ('DW_FORM_data1', 'DW_FORM_data2', 'DW_FORM_data4', 'DW_FORM_data8') and (not attr.name == 'DW_AT_const_value') or attr.form in ('DW_FORM_sec_offset', 'DW_FORM_loclistx')) and (not LocationParser._attribute_is_constant(attr, dwarf_version))

    @staticmethod
    def _attribute_is_constant(attr, dwarf_version):
        return (dwarf_version >= 3 and attr.name == 'DW_AT_data_member_location' or attr.name in ('DW_AT_upper_bound', 'DW_AT_count')) and attr.form in ('DW_FORM_data1', 'DW_FORM_data2', 'DW_FORM_data4', 'DW_FORM_data8', 'DW_FORM_sdata', 'DW_FORM_udata')

    @staticmethod
    def _attribute_is_loclistptr_class(attr):
        return attr.name in ('DW_AT_location', 'DW_AT_string_length', 'DW_AT_const_value', 'DW_AT_return_addr', 'DW_AT_data_member_location', 'DW_AT_frame_base', 'DW_AT_segment', 'DW_AT_static_link', 'DW_AT_use_location', 'DW_AT_vtable_elem_location', 'DW_AT_call_value', 'DW_AT_GNU_call_site_value', 'DW_AT_GNU_call_site_target', 'DW_AT_GNU_call_site_data_value', 'DW_AT_call_target', 'DW_AT_call_target_clobbered', 'DW_AT_call_data_location', 'DW_AT_call_data_value', 'DW_AT_upper_bound', 'DW_AT_count')
from __future__ import (absolute_import, division, print_function)
import abc
import os
import json
import subprocess
from ansible.plugins.lookup import LookupBase
from ansible.errors import AnsibleLookupError, AnsibleOptionsError
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.module_utils.six import with_metaclass
from ansible_collections.community.general.plugins.module_utils.onepassword import OnePasswordConfig
def _parse_field(self, data_json, field_name, section_title=None):
    """
        Schema reference: https://developer.1password.com/docs/cli/item-template-json

        Example Data:

            # Password item
            {
              "id": "ywvdbojsguzgrgnokmcxtydgdv",
              "title": "Authy Backup",
              "version": 1,
              "vault": {
                "id": "bcqxysvcnejjrwzoqrwzcqjqxc",
                "name": "Personal"
              },
              "category": "PASSWORD",
              "last_edited_by": "7FUPZ8ZNE02KSHMAIMKHIVUE17",
              "created_at": "2015-01-18T13:13:38Z",
              "updated_at": "2016-02-20T16:23:54Z",
              "additional_information": "Jan 18, 2015, 08:13:38",
              "fields": [
                {
                  "id": "password",
                  "type": "CONCEALED",
                  "purpose": "PASSWORD",
                  "label": "password",
                  "value": "OctoberPoppyNuttyDraperySabbath",
                  "reference": "op://Personal/Authy Backup/password",
                  "password_details": {
                    "strength": "FANTASTIC"
                  }
                },
                {
                  "id": "notesPlain",
                  "type": "STRING",
                  "purpose": "NOTES",
                  "label": "notesPlain",
                  "value": "Backup password to restore Authy",
                  "reference": "op://Personal/Authy Backup/notesPlain"
                }
              ]
            }

            # Login item
            {
              "id": "awk4s2u44fhnrgppszcsvc663i",
              "title": "Dummy Login",
              "version": 2,
              "vault": {
                "id": "stpebbaccrq72xulgouxsk4p7y",
                "name": "Personal"
              },
              "category": "LOGIN",
              "last_edited_by": "LSGPJERUYBH7BFPHMZ2KKGL6AU",
              "created_at": "2018-04-25T21:55:19Z",
              "updated_at": "2018-04-25T21:56:06Z",
              "additional_information": "agent.smith",
              "urls": [
                {
                  "primary": true,
                  "href": "https://acme.com"
                }
              ],
              "sections": [
                {
                  "id": "linked items",
                  "label": "Related Items"
                }
              ],
              "fields": [
                {
                  "id": "username",
                  "type": "STRING",
                  "purpose": "USERNAME",
                  "label": "username",
                  "value": "agent.smith",
                  "reference": "op://Personal/Dummy Login/username"
                },
                {
                  "id": "password",
                  "type": "CONCEALED",
                  "purpose": "PASSWORD",
                  "label": "password",
                  "value": "Q7vFwTJcqwxKmTU]Dzx7NW*wrNPXmj",
                  "entropy": 159.6083697084228,
                  "reference": "op://Personal/Dummy Login/password",
                  "password_details": {
                    "entropy": 159,
                    "generated": true,
                    "strength": "FANTASTIC"
                  }
                },
                {
                  "id": "notesPlain",
                  "type": "STRING",
                  "purpose": "NOTES",
                  "label": "notesPlain",
                  "reference": "op://Personal/Dummy Login/notesPlain"
                }
              ]
            }
        """
    data = json.loads(data_json)
    field_name = _lower_if_possible(field_name)
    for field in data.get('fields', []):
        if section_title is None:
            if field.get(field_name):
                return field.get(field_name)
            if field.get('label', '').lower() == field_name:
                return field.get('value', '')
            if field.get('id', '').lower() == field_name:
                return field.get('value', '')
        section = field.get('section', {})
        section_title = _lower_if_possible(section_title)
        current_section_title = section.get('label', section.get('id', '')).lower()
        if section_title == current_section_title:
            if field.get('label', '').lower() == field_name:
                return field.get('value', '')
            if field.get('id', '').lower() == field_name:
                return field.get('value', '')
    return ''
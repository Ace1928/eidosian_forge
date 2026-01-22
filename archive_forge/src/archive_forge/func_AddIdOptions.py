from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def AddIdOptions(parser, entity, plural_entity, action_description):
    parser.add_argument('ids', metavar='ID', nargs='*', help='          Zero or more {entity} resource identifiers. The specified\n          {plural_entity} will be {action_description}.\n      '.format(entity=entity, plural_entity=plural_entity, action_description=action_description))
    parser.add_argument('--location', metavar='LOCATION-REGEXP', action='append', help='          A regular expression to match against {entity}\n          locations. All {plural_entity} matching this value will be\n          {action_description}.  You may specify --location multiple times.\n\n          EXAMPLE:\n\n            {{command}} \\\n                --location foo.py:[1-3] --location bar.py:4\n      '.format(entity=entity, plural_entity=plural_entity, action_description=action_description))
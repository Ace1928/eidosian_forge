from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import os.path
from googlecloudsdk.core import branding
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import name_parsing
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import files
from mako import runtime
from mako import template
def _BuildCommandGroupInitContext(collection_info, release_tracks, resource_data):
    """Makes context dictionary for config init file template rendering."""
    init_dict = {}
    init_dict['utf_encoding'] = '-*- coding: utf-8 -*- #'
    init_dict['current_year'] = datetime.datetime.now().year
    init_dict['branded_api_name'] = branding.Branding().get(collection_info.api_name, collection_info.api_name.capitalize())
    init_dict['singular_resource_name_with_spaces'] = name_parsing.convert_collection_name_to_delimited(collection_info.name)
    release_track_string = ''
    for x, release_track in enumerate(release_tracks):
        release_track_string += 'base.ReleaseTrack.{}'.format(release_track.upper())
        if x != len(release_tracks) - 1:
            release_track_string += ', '
    init_dict['release_tracks'] = release_track_string
    if 'group_category' in resource_data:
        init_dict['group_category'] = resource_data.group_category
    return init_dict
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.command_lib.container.fleet.features import base
def ModeEnumTranslation(mode) -> str:
    if mode == 'unspecified':
        return 'ACTUATION_MODE_UNSPECIFIED'
    if mode == 'create-and-delete-if-created':
        return 'ACTUATION_MODE_CREATE_AND_DELETE_IF_CREATED'
    if mode == 'add-and-remove-fleet-labels':
        return 'ACTUATION_MODE_ADD_AND_REMOVE_FLEET_LABELS'
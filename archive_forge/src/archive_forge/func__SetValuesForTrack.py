from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from googlecloudsdk.calliope import base
def _SetValuesForTrack(obj, track):
    """Recursively modify an object to have only values for the provided track.

  Args:
    obj: The object to modify.
    track: The track to extract the values for.
  Returns:
    The modified object
  Raises:
    DoesNotExistForTrackError: if the object does not exist for the track.
  """
    if isinstance(obj, dict):
        is_group = GROUP in obj
        if RELEASE_TRACKS in obj:
            if track not in obj[RELEASE_TRACKS]:
                raise DoesNotExistForTrackError()
            del obj[RELEASE_TRACKS]
        if track in obj:
            for key, value in obj[track].items():
                obj[key] = value
        for track_key in ALL_TRACKS:
            if track_key in obj:
                del obj[track_key]
        for key, child in list(obj.items()):
            try:
                _SetValuesForTrack(child, track)
            except DoesNotExistForTrackError:
                del obj[key]
        if is_group and (not obj):
            raise DoesNotExistForTrackError()
    elif isinstance(obj, list):
        children = list(obj)
        obj[:] = []
        for child in children:
            try:
                obj.append(_SetValuesForTrack(child, track))
            except DoesNotExistForTrackError:
                pass
    return obj
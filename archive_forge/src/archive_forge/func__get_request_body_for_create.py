import warnings
from novaclient import api_versions
from novaclient import base
@staticmethod
def _get_request_body_for_create(volume_id, device=None, tag=None, delete_on_termination=False):
    body = {'volumeAttachment': {'volumeId': volume_id}}
    if device is not None:
        body['volumeAttachment']['device'] = device
    if tag is not None:
        body['volumeAttachment']['tag'] = tag
    if delete_on_termination:
        body['volumeAttachment']['delete_on_termination'] = delete_on_termination
    return body
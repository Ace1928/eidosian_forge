from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def CreateOperationQuotaExceededMsg(data):
    """Constructs message to show for quota exceeded error."""
    error_info = None
    localized_message = None
    try:
        error = data.get('error')
        for item in error.get('details'):
            if item.get('@type') == 'type.googleapis.com/google.rpc.ErrorInfo':
                error_info = item
            if item.get('@type') == 'type.googleapis.com/google.rpc.LocalizedMessage':
                localized_message = item
        localized_message_text = localized_message.get('message')
        metadata = error_info.get('metadatas')
        container_type = metadata.get('containerType')
        container_id = metadata.get('containerId')
        location = metadata.get('location')
        if None in (localized_message_text, container_type, container_id, location):
            return error.get('message')
        return '{}\n{}\n\tcontainer type = {}\n\tcontainer id = {}\n\tlocation = {}\nWait for other operations to be done, or view documentation on best practices for reducing concurrent operations: https://cloud.google.com/compute/quotas#best_practices.'.format(error.get('message'), localized_message_text, container_type, container_id, location)
    except (KeyError, AttributeError):
        return error.get('message')
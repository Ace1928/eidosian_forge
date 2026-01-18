import logging
import uuid
def validate_ref_and_return_uuid(ref, entity):
    """Verifies that there is a real uuid (possibly at the end of a uri)

    :return: The uuid.UUID object
    :raises ValueError: If it cannot correctly parse the uuid in the ref.
    """
    try:
        ref_pieces = ref.rstrip('/').rsplit('/', 1)
        return uuid.UUID(ref_pieces[-1])
    except Exception:
        raise ValueError('{0} incorrectly specified.'.format(entity))
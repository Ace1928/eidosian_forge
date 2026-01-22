from .repository import SmartServerRepositoryRequest
from .request import SuccessfulSmartServerResponse
class SmartServerPackRepositoryAutopack(SmartServerRepositoryRequest):

    def do_repository_request(self, repository):
        pack_collection = getattr(repository, '_pack_collection', None)
        if pack_collection is None:
            return SuccessfulSmartServerResponse((b'ok',))
        with repository.lock_write():
            repository._pack_collection.autopack()
        return SuccessfulSmartServerResponse((b'ok',))
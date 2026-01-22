from .object_store import iter_tree_contents
from .objects import Blob
from .patch import is_binary
class TreeBlobNormalizer(BlobNormalizer):

    def __init__(self, config_stack, git_attributes, object_store, tree=None) -> None:
        super().__init__(config_stack, git_attributes)
        if tree:
            self.existing_paths = {name for name, _, _ in iter_tree_contents(object_store, tree)}
        else:
            self.existing_paths = set()

    def checkin_normalize(self, blob, tree_path):
        if self.fallback_read_filter is not None or tree_path not in self.existing_paths:
            return super().checkin_normalize(blob, tree_path)
        return blob
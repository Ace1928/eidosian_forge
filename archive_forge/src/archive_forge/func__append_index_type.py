from enum import Enum
def _append_index_type(self, index_type):
    """Append `ON HASH` or `ON JSON` according to the enum."""
    if index_type is IndexType.HASH:
        self.args.extend(['ON', 'HASH'])
    elif index_type is IndexType.JSON:
        self.args.extend(['ON', 'JSON'])
    elif index_type is not None:
        raise RuntimeError(f'index_type must be one of {list(IndexType)}')
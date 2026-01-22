from io import BytesIO
from . import osutils, progress, trace
from .i18n import gettext
from .ui import ui_factory
@staticmethod
def _match_hits(hit_list):
    """Using a hit list, determine a path-to-fileid map.

        The hit list is a list of (count, path, file_id), where count is a
        (possibly float) number, with higher numbers indicating stronger
        matches.
        """
    seen_file_ids = set()
    path_map = {}
    for count, path, file_id in sorted(hit_list, reverse=True):
        if path in path_map or file_id in seen_file_ids:
            continue
        path_map[path] = file_id
        seen_file_ids.add(file_id)
    return path_map
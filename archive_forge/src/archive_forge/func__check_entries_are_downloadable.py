import filecmp
import logging
import os
import requests
import wandb
def _check_entries_are_downloadable(art):
    entries = _collect_entries(art)
    for entry in entries:
        if not _check_entry_is_downloable(entry):
            return False
    return True
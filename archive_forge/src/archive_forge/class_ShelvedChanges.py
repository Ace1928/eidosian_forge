class ShelvedChanges(UncommittedChanges):
    _fmt = 'Working tree "%(display_url)s" has shelved changes (See brz shelve --list).%(more)s'
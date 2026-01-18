import urllib.parse
Extract Mercurial metadata from a commit message.

    :param message: Commit message to extract from
    :return: Tuple with original commit message, renames, branch and
        extra data.
    
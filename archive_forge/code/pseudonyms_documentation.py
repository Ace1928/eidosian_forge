from collections import defaultdict
from breezy import errors, foreign, ui, urlutils
Create a rebase map from pseudonyms and existing/desired ancestry.

    :param pseudonym_dict: Dictionary with pseudonym as returned by
        pseudonyms_as_dict()
    :param existing: Existing ancestry, might need to be rebased
    :param desired: Desired ancestry
    :return: rebase map, as dictionary
    
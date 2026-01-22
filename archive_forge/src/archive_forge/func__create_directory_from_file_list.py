from typing import Dict, List
from .glob_group import GlobGroup, GlobPattern
def _create_directory_from_file_list(filename: str, file_list: List[str], include: 'GlobPattern'='**', exclude: 'GlobPattern'=()) -> Directory:
    """Return a :class:`Directory` file structure representation created from a list of files.

    Args:
        filename (str): The name given to the top-level directory that will be the
            relative root for all file paths found in the file_list.

        file_list (List[str]): List of files to add to the top-level directory.

        include (Union[List[str], str]): An optional pattern that limits what is included from the file_list to
            files whose name matches the pattern.

        exclude (Union[List[str], str]): An optional pattern that excludes files whose name match the pattern.

    Returns:
            :class:`Directory`: a :class:`Directory` file structure representation created from a list of files.
    """
    glob_pattern = GlobGroup(include, exclude=exclude, separator='/')
    top_dir = Directory(filename, True)
    for file in file_list:
        if glob_pattern.matches(file):
            top_dir._add_file(file)
    return top_dir
from typing import TYPE_CHECKING, Iterator, List, Optional, Tuple, Union
Parse a string referring to a single commit.

    Args:
      repo: A` Repo` object
      committish: A string referring to a single commit.
    Returns: A Commit object
    Raises:
      KeyError: When the reference commits can not be found
      ValueError: If the range can not be parsed
    
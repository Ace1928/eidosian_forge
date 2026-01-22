import difflib
import typing
import click
class DYMCommandCollection(DYMMixin, click.CommandCollection):
    """
    click CommandCollection to provide git-like
    *did-you-mean* functionality when a certain
    command is not found in the group.
    """
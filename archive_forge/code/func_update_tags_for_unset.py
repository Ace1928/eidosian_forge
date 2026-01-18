import argparse
from osc_lib.i18n import _
def update_tags_for_unset(client, obj, parsed_args):
    """Unset the tags on an object.

    :param client: The service client to use unsetting the tags.
    :param obj: The object (Resource) to unset the tags on.
    :param parsed_args: Parsed argument object returned by argparse parse_args.
    """
    tags = set(obj.tags)
    if parsed_args.all_tag:
        tags = set()
    if parsed_args.tags:
        tags -= set(parsed_args.tags)
    if set(obj.tags) != tags:
        client.set_tags(obj, sorted(list(tags)))
def _write_kegg(item, info, indent=KEGG_ITEM_LENGTH):
    """Write a indented KEGG record item (PRIVATE).

    Arguments:
     - item - The name of the item to be written.
     - info - The (wrapped) information to write.
     - indent - Width of item field.

    """
    s = ''
    for line in info:
        partial_lines = line.splitlines()
        for partial in partial_lines:
            s += item.ljust(indent) + partial + '\n'
            if item:
                item = ''
    return s
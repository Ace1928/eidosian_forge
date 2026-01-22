def _get_noqa_for_var(prop_name):
    return '  # noqa (assign to builtin)' if prop_name in ('type', 'format', 'id', 'hex', 'breakpoint', 'filter') else ''
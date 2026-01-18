Parse a DJANGO_COLORS environment variable to produce the system palette

    The general form of a palette definition is:

        "palette;role=fg;role=fg/bg;role=fg,option,option;role=fg/bg,option,option"

    where:
        palette is a named palette; one of 'light', 'dark', or 'nocolor'.
        role is a named style used by Django
        fg is a foreground color.
        bg is a background color.
        option is a display options.

    Specifying a named palette is the same as manually specifying the individual
    definitions for each role. Any individual definitions following the palette
    definition will augment the base palette definition.

    Valid roles:
        'error', 'success', 'warning', 'notice', 'sql_field', 'sql_coltype',
        'sql_keyword', 'sql_table', 'http_info', 'http_success',
        'http_redirect', 'http_not_modified', 'http_bad_request',
        'http_not_found', 'http_server_error', 'migrate_heading',
        'migrate_label'

    Valid colors:
        'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'

    Valid options:
        'bold', 'underscore', 'blink', 'reverse', 'conceal', 'noreset'
    
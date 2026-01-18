from __future__ import absolute_import, print_function, division
from petl.util.base import Table, iterdata
from petl.compat import text_type
from petl.errors import ArgumentError as PetlArgError
def togsheet(table, credentials_or_client, spreadsheet, worksheet=None, cell_range=None, share_emails=None, role='reader'):
    """
    Write a table to a new google sheet.

    The `credentials_or_client` are used to authenticate with the google apis.
    For more info, check `authentication`_. 

    The `spreadsheet` will be the title of the workbook created google sheets.
    If there is a spreadsheet with same title a new one will be created.

    If `worksheet` is specified, the first worksheet in the spreadsheet
    will be renamed to its value.

    The spreadsheet will be shared with all emails in `share_emails` with
    `role` permissions granted. For more info, check `sharing`_. 

    Returns: the spreadsheet key that can be used in `appendgsheet` further.


    .. _sharing: https://developers.google.com/drive/v3/web/manage-sharing

    .. note::
        The `gspread`_ package doesn't support serialization of `date` and 
        `datetime` types yet.

    Example usage::

        >>> from petl import fromcolumns, togsheet
        >>> import gspread # doctest: +SKIP
        >>> client = gspread.service_account() # doctest: +SKIP
        >>> cols = [[0, 1, 2], ['a', 'b', 'c']]
        >>> tbl = fromcolumns(cols)
        >>> togsheet(tbl, client, 'example_spreadsheet') # doctest: +SKIP
    """
    gspread_client = _get_gspread_client(credentials_or_client)
    wb = gspread_client.create(spreadsheet)
    ws = wb.sheet1
    ws.resize(rows=1, cols=1)
    if worksheet is not None:
        ws.update_title(title=worksheet)
    ws.append_rows(table, table_range=cell_range)
    if share_emails is not None:
        for user_email in share_emails:
            wb.share(user_email, perm_type='user', role=role)
    return wb.id
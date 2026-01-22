import sys
def add_notice_to_docstring(doc, instructions, no_doc_str, suffix_str, notice, notice_type='Warning'):
    """Adds a deprecation notice to a docstring.

  Args:
    doc: The original docstring.
    instructions: A string, describing how to fix the problem.
    no_doc_str: The default value to use for `doc` if `doc` is empty.
    suffix_str: Is added to the end of the first line.
    notice: A list of strings. The main notice warning body.
    notice_type: The type of notice to use. Should be one of `[Caution,
    Deprecated, Important, Note, Warning]`

  Returns:
    A new docstring, with the notice attached.

  Raises:
    ValueError: If `notice` is empty.
  """
    allowed_notice_types = ['Deprecated', 'Warning', 'Caution', 'Important', 'Note']
    if notice_type not in allowed_notice_types:
        raise ValueError(f'Unrecognized notice type. Should be one of: {allowed_notice_types}')
    if not doc:
        lines = [no_doc_str]
    else:
        lines = _normalize_docstring(doc).splitlines()
        lines[0] += ' ' + suffix_str
    if not notice:
        raise ValueError('The `notice` arg must not be empty.')
    notice[0] = f'{notice_type}: {notice[0]}'
    notice = [''] + notice + ([instructions] if instructions else [])
    if len(lines) > 1:
        if lines[1].strip():
            notice.append('')
        lines[1:1] = notice
    else:
        lines += notice
    return '\n'.join(lines)
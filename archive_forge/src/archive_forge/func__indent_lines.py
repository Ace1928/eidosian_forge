def _indent_lines(lines, indent='    '):
    out_lines = []
    for line in lines.splitlines(keepends=True):
        out_lines.append(indent + line)
    return ''.join(out_lines)
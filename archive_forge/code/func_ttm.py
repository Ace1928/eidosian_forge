import subprocess
def ttm(math_code, reporter=None):
    """Convert LaTeX math code to MathML with TtM_

    .. _TtM: http://hutchinson.belmont.ma.us/tth/mml/
    """
    p = subprocess.Popen(['ttm', '-u', '-r'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True)
    p.stdin.write((document_template % math_code).encode('utf8'))
    p.stdin.close()
    result = p.stdout.read()
    err = p.stderr.read().decode('utf8')
    if err.find('**** Unknown') >= 0:
        msg = '\n'.join([line for line in err.splitlines() if line.startswith('****')])
        raise SyntaxError('\nMessage from external converter TtM:\n' + msg)
    if reporter and err.find('**** Error') >= 0 or not result:
        reporter.error(err)
    start, end = (result.find('<math'), result.find('</math>') + 7)
    result = result[start:end]
    return result
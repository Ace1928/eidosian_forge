import os
def _permissions():
    try:
        if not os.path.exists(PLOTLY_DIR):
            try:
                os.mkdir(PLOTLY_DIR)
            except Exception:
                if not os.path.isdir(PLOTLY_DIR):
                    raise
        with open(TEST_FILE, 'w') as f:
            f.write('testing\n')
        try:
            os.remove(TEST_FILE)
        except Exception:
            pass
        return True
    except Exception:
        return False
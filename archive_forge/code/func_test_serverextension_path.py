import os
def test_serverextension_path(app):
    import jupyter_lsp
    paths = jupyter_lsp._jupyter_server_extension_paths()
    for path in paths:
        assert __import__(path['module'])
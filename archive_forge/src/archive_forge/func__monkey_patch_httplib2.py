import os
import pkgutil
import shutil
import tempfile
import httplib2
def _monkey_patch_httplib2(extract_dir):
    """Patch things so that httplib2 works properly in a PAR.

  Manually extract certificates to file to make OpenSSL happy and avoid error:
     ssl.SSLError: [Errno 185090050] _ssl.c:344: error:0B084002:x509 ...

  Args:
    extract_dir: the directory into which we extract the necessary files.
  """
    if os.path.isfile(httplib2.CA_CERTS):
        return
    cacerts_contents = pkgutil.get_data('httplib2', 'cacerts.txt')
    cacerts_filename = os.path.join(extract_dir, 'cacerts.txt')
    with open(cacerts_filename, 'wb') as f:
        f.write(cacerts_contents)
    httplib2.CA_CERTS = cacerts_filename
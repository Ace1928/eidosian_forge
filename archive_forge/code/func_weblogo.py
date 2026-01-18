from urllib.parse import urlencode
from urllib.request import urlopen, Request
import warnings
from Bio import BiopythonDeprecationWarning
from Bio.Align import Alignment
from Bio.Seq import reverse_complement
def weblogo(self, fname, fmt='PNG', version='2.8.2', **kwds):
    """Download and save a weblogo using the Berkeley weblogo service.

        Requires an internet connection.

        The parameters from ``**kwds`` are passed directly to the weblogo server.

        Currently, this method uses WebLogo version 3.3.
        These are the arguments and their default values passed to
        WebLogo 3.3; see their website at http://weblogo.threeplusone.com
        for more information::

            'stack_width' : 'medium',
            'stacks_per_line' : '40',
            'alphabet' : 'alphabet_dna',
            'ignore_lower_case' : True,
            'unit_name' : "bits",
            'first_index' : '1',
            'logo_start' : '1',
            'logo_end': str(self.length),
            'composition' : "comp_auto",
            'percentCG' : '',
            'scale_width' : True,
            'show_errorbars' : True,
            'logo_title' : '',
            'logo_label' : '',
            'show_xaxis': True,
            'xaxis_label': '',
            'show_yaxis': True,
            'yaxis_label': '',
            'yaxis_scale': 'auto',
            'yaxis_tic_interval' : '1.0',
            'show_ends' : True,
            'show_fineprint' : True,
            'color_scheme': 'color_auto',
            'symbols0': '',
            'symbols1': '',
            'symbols2': '',
            'symbols3': '',
            'symbols4': '',
            'color0': '',
            'color1': '',
            'color2': '',
            'color3': '',
            'color4': '',

        """
    if set(self.alphabet) == set('ACDEFGHIKLMNPQRSTVWY'):
        alpha = 'alphabet_protein'
    elif set(self.alphabet) == set('ACGU'):
        alpha = 'alphabet_rna'
    elif set(self.alphabet) == set('ACGT'):
        alpha = 'alphabet_dna'
    else:
        alpha = 'auto'
    frequencies = format(self, 'transfac')
    url = 'https://weblogo.threeplusone.com/create.cgi'
    values = {'sequences': frequencies, 'format': fmt.lower(), 'stack_width': 'medium', 'stacks_per_line': '40', 'alphabet': alpha, 'ignore_lower_case': True, 'unit_name': 'bits', 'first_index': '1', 'logo_start': '1', 'logo_end': str(self.length), 'composition': 'comp_auto', 'percentCG': '', 'scale_width': True, 'show_errorbars': True, 'logo_title': '', 'logo_label': '', 'show_xaxis': True, 'xaxis_label': '', 'show_yaxis': True, 'yaxis_label': '', 'yaxis_scale': 'auto', 'yaxis_tic_interval': '1.0', 'show_ends': True, 'show_fineprint': True, 'color_scheme': 'color_auto', 'symbols0': '', 'symbols1': '', 'symbols2': '', 'symbols3': '', 'symbols4': '', 'color0': '', 'color1': '', 'color2': '', 'color3': '', 'color4': ''}
    values.update({k: '' if v is False else str(v) for k, v in kwds.items()})
    data = urlencode(values).encode('utf-8')
    req = Request(url, data)
    response = urlopen(req)
    with open(fname, 'wb') as f:
        im = response.read()
        f.write(im)
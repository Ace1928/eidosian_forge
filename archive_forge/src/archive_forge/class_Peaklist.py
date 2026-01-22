class Peaklist:
    """Provide access to header lines and data from a nmrview xpk file.

    Header file lines and file data are available as attributes.

    Parameters
    ----------
    infn : str
        The input nmrview filename.

    Attributes
    ----------
    firstline  : str
        The first line in the header.
    axislabels : str
        The axis labels.
    dataset    : str
        The label of the dataset.
    sw         : str
        The sw coordinates.
    sf         : str
        The sf coordinates.
    datalabels : str
        The labels of the entries.

    data : list
        File data after header lines.

    Examples
    --------
    >>> from Bio.NMR.xpktools import Peaklist
    >>> peaklist = Peaklist('../Doc/examples/nmr/noed.xpk')
    >>> peaklist.firstline
    'label dataset sw sf '
    >>> peaklist.dataset
    'test.nv'
    >>> peaklist.sf
    '{599.8230 } { 60.7860 } { 60.7860 }'
    >>> peaklist.datalabels
    ' H1.L  H1.P  H1.W  H1.B  H1.E  H1.J  15N2.L  15N2.P  15N2.W  15N2.B  15N2.E  15N2.J  N15.L  N15.P  N15.W  N15.B  N15.E  N15.J  vol  int  stat '

    """

    def __init__(self, infn):
        """Initialize the class."""
        with open(infn) as infile:
            self.firstline = infile.readline().split('\n')[0]
            self.axislabels = infile.readline().split('\n')[0]
            self.dataset = infile.readline().split('\n')[0]
            self.sw = infile.readline().split('\n')[0]
            self.sf = infile.readline().split('\n')[0]
            self.datalabels = infile.readline().split('\n')[0]
            self.data = [line.split('\n')[0] for line in infile]

    def residue_dict(self, index):
        """Return a dict of lines in 'data' indexed by residue number or a nucleus.

        The nucleus should be given as the input argument in the same form as
        it appears in the xpk label line (H1, 15N for example)

        Parameters
        ----------
        index : str
            The nucleus to index data by.

        Returns
        -------
        resdict : dict
            Mappings of index nucleus to data line.

        Examples
        --------
        >>> from Bio.NMR.xpktools import Peaklist
        >>> peaklist = Peaklist('../Doc/examples/nmr/noed.xpk')
        >>> residue_d = peaklist.residue_dict('H1')
        >>> sorted(residue_d.keys())
        ['10', '3', '4', '5', '6', '7', '8', '9', 'maxres', 'minres']
        >>> residue_d['10']
        ['8  10.hn   7.663   0.021   0.010   ++   0.000   10.n   118.341   0.324   0.010   +E   0.000   10.n   118.476   0.324   0.010   +E   0.000  0.49840 0.49840 0']

        """
        maxres = -1
        minres = -1
        self.dict = {}
        for line in self.data:
            ind = XpkEntry(line, self.datalabels).fields[index + '.L']
            key = ind.split('.')[0]
            res = int(key)
            if maxres == -1:
                maxres = res
            if minres == -1:
                minres = res
            maxres = max([maxres, res])
            minres = min([minres, res])
            res = str(res)
            try:
                self.dict[res].append(line)
            except KeyError:
                self.dict[res] = [line]
        self.dict['maxres'] = maxres
        self.dict['minres'] = minres
        return self.dict

    def write_header(self, outfn):
        """Write header lines from input file to handle ``outfn``."""
        with open(outfn, 'w') as outfile:
            outfile.write(self.firstline)
            outfile.write('\n')
            outfile.write(self.axislabels)
            outfile.write('\n')
            outfile.write(self.dataset)
            outfile.write('\n')
            outfile.write(self.sw)
            outfile.write('\n')
            outfile.write(self.sf)
            outfile.write('\n')
            outfile.write(self.datalabels)
            outfile.write('\n')
import logging
from rdkit import RDLogger
from rdkit.Chem import rdinchi
def MolToInchi(mol, options='', logLevel=None, treatWarningAsError=False):
    """Returns the standard InChI string for a molecule

    Keyword arguments:
    logLevel -- the log level used for logging logs and messages from InChI
    API. set to None to diable the logging completely
    treatWarningAsError -- set to True to raise an exception in case of a
    molecule that generates warning in calling InChI API. The resultant InChI
    string and AuxInfo string as well as the error message are encoded in the
    exception.

    Returns:
    the standard InChI string returned by InChI API for the input molecule
    """
    if options.find('AuxNone') == -1:
        if options:
            options += ' /AuxNone'
        else:
            options += '/AuxNone'
    try:
        inchi, aux = MolToInchiAndAuxInfo(mol, options, logLevel=logLevel, treatWarningAsError=treatWarningAsError)
    except InchiReadWriteError as inst:
        inchi, aux, message = inst.args
        raise InchiReadWriteError(inchi, message)
    return inchi
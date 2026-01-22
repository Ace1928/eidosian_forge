import logging
import sys
from base64 import b64encode
import numpy as np
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, SDWriter, rdchem
from rdkit.Chem.Scaffolds import MurckoScaffold
from io import BytesIO
from xml.dom import minidom
def WriteSDF(df, out, molColName='ROMol', idName=None, properties=None, allNumeric=False, forceV3000=False):
    """Write an SD file for the molecules in the dataframe. Dataframe columns can be exported as
    SDF tags if specified in the "properties" list. "properties=list(df.columns)" would export
    all columns.
    The "allNumeric" flag allows to automatically include all numeric columns in the output.
    User has to make sure that correct data type is assigned to column.
    "idName" can be used to select a column to serve as molecule title. It can be set to
    "RowID" to use the dataframe row key as title.
    """
    close = None
    if isinstance(out, str):
        if out.lower()[-3:] == '.gz':
            import gzip
            out = gzip.open(out, 'wt')
            close = out.close
    writer = SDWriter(out)
    if forceV3000:
        writer.SetForceV3000(True)
    if properties is None:
        properties = []
    else:
        properties = list(properties)
    if allNumeric:
        properties.extend([dt for dt in df.dtypes.keys() if np.issubdtype(df.dtypes[dt], np.floating) or np.issubdtype(df.dtypes[dt], np.integer)])
    if molColName in properties:
        properties.remove(molColName)
    if idName in properties:
        properties.remove(idName)
    writer.SetProps(properties)
    for row in df.iterrows():
        mol = Chem.Mol(row[1][molColName])
        if idName is not None:
            if idName == 'RowID':
                mol.SetProp('_Name', str(row[0]))
            else:
                mol.SetProp('_Name', str(row[1][idName]))
        for p in properties:
            cell_value = row[1][p]
            if np.issubdtype(type(cell_value), np.floating):
                s = '{:f}'.format(cell_value).rstrip('0')
                if s[-1] == '.':
                    s += '0'
                mol.SetProp(p, s)
            else:
                mol.SetProp(p, str(cell_value))
        writer.write(mol)
    writer.close()
    if close is not None:
        close()
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
def SaveXlsxFromFrame(frame, outFile, molCol='ROMol', size=(300, 300), formats=None):
    """
      Saves pandas DataFrame as a xlsx file with embedded images.
      molCol can be either a single column label or a list of column labels.
      It maps numpy data types to excel cell types:
      int, float -> number
      datetime -> datetime
      object -> string (limited to 32k character - xlsx limitations)

      The formats parameter can be optionally set to a dict of XlsxWriter
      formats (https://xlsxwriter.readthedocs.io/format.html#format), e.g.:
      {
        'write_string':  {'text_wrap': True}
      }
      Currently supported keys for the formats dict are:
      'write_string', 'write_number', 'write_datetime'.

      Cells with compound images are a bit larger than images due to excel.
      Column width weirdness explained (from xlsxwriter docs):
      The width corresponds to the column width value that is specified in Excel.
      It is approximately equal to the length of a string in the default font of Calibri 11.
      Unfortunately, there is no way to specify "AutoFit" for a column in the Excel file format.
      This feature is only available at runtime from within Excel.
      """
    import xlsxwriter
    cols = list(frame.columns)
    if isinstance(molCol, str):
        molCol = [molCol]
    molCol = list(set(molCol))
    dataTypes = dict(frame.dtypes)
    molCol_indices = [cols.index(mc) for mc in molCol]
    workbook = xlsxwriter.Workbook(outFile)
    cell_formats = {}
    formats = formats or {}
    for key in ['write_string', 'write_number', 'write_datetime']:
        format = formats.get(key, None)
        if format is not None:
            format = workbook.add_format(format)
        cell_formats[key] = format
    worksheet = workbook.add_worksheet()
    for col_idx, col in enumerate(cols):
        worksheet.write_string(0, col_idx, col)
    for row_idx, (_, row) in enumerate(frame.iterrows()):
        row_idx_actual = row_idx + 1
        worksheet.set_row(row_idx_actual, height=size[1])
        for col_idx, col in enumerate(cols):
            if col_idx in molCol_indices:
                image_data = BytesIO()
                m = row[col]
                img = Draw.MolToImage(m if isinstance(m, Chem.Mol) else Chem.Mol(), size=size, options=drawOptions)
                img.save(image_data, format='PNG')
                worksheet.insert_image(row_idx_actual, col_idx, 'f', {'image_data': image_data})
                worksheet.set_column(col_idx, col_idx, width=size[0] / 6.0)
            elif str(dataTypes[col]) == 'object':
                worksheet.write_string(row_idx_actual, col_idx, str(row[col])[:32000], cell_formats['write_string'])
            elif 'float' in str(dataTypes[col]) or 'int' in str(dataTypes[col]):
                if row[col] != np.nan or row[col] != np.inf:
                    worksheet.write_number(row_idx_actual, col_idx, row[col], cell_formats['write_number'])
            elif 'datetime' in str(dataTypes[col]):
                worksheet.write_datetime(row_idx_actual, col_idx, row[col], cell_formats['write_datetime'])
    workbook.close()
    image_data.close()
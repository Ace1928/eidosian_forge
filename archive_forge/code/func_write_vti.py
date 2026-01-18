import numpy as np
def write_vti(filename, atoms, data=None):
    from vtk import vtkStructuredPoints, vtkDoubleArray, vtkXMLImageDataWriter
    if isinstance(atoms, list):
        if len(atoms) > 1:
            raise ValueError('Can only write one configuration to a VTI file!')
        atoms = atoms[0]
    if data is None:
        raise ValueError('VTK XML Image Data (VTI) format requires data!')
    data = np.asarray(data)
    if data.dtype == complex:
        data = np.abs(data)
    cell = atoms.get_cell()
    if not np.all(cell == np.diag(np.diag(cell))):
        raise ValueError('Unit cell must be orthogonal')
    bbox = np.array(list(zip(np.zeros(3), cell.diagonal()))).ravel()
    spts = vtkStructuredPoints()
    spts.SetWholeBoundingBox(bbox)
    spts.SetDimensions(data.shape)
    spts.SetSpacing(cell.diagonal() / data.shape)
    da = vtkDoubleArray()
    da.SetName('scalars')
    da.SetNumberOfComponents(1)
    da.SetNumberOfTuples(np.prod(data.shape))
    for i, d in enumerate(data.swapaxes(0, 2).flatten()):
        da.SetTuple1(i, d)
    spd = spts.GetPointData()
    spd.SetScalars(da)
    "\n    from vtk.util.vtkImageImportFromArray import vtkImageImportFromArray\n    iia = vtkImageImportFromArray()\n    #iia.SetArray(Numeric_asarray(data.swapaxes(0,2).flatten()))\n    iia.SetArray(Numeric_asarray(data))\n    ida = iia.GetOutput()\n    ipd = ida.GetPointData()\n    ipd.SetName('scalars')\n    spd.SetScalars(ipd.GetScalars())\n    "
    w = vtkXMLImageDataWriter()
    if fast:
        w.SetDataModeToAppend()
        w.EncodeAppendedDataOff()
    else:
        w.SetDataModeToAscii()
    w.SetFileName(filename)
    w.SetInput(spts)
    w.Write()
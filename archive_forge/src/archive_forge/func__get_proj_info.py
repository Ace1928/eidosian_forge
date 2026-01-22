import importlib.metadata
import platform
import sys
def _get_proj_info():
    """Information on system PROJ

    Returns
    -------
    proj_info: dict
        system PROJ information
    """
    import pyproj
    from pyproj.database import get_database_metadata
    from pyproj.exceptions import DataDirError
    try:
        data_dir = pyproj.datadir.get_data_dir()
    except DataDirError:
        data_dir = None
    blob = [('pyproj', pyproj.__version__), ('PROJ', pyproj.__proj_version__), ('data dir', data_dir), ('user_data_dir', pyproj.datadir.get_user_data_dir()), ('PROJ DATA (recommended version)', get_database_metadata('PROJ_DATA.VERSION')), ('PROJ Database', f'{get_database_metadata('DATABASE.LAYOUT.VERSION.MAJOR')}.{get_database_metadata('DATABASE.LAYOUT.VERSION.MINOR')}'), ('EPSG Database', f'{get_database_metadata('EPSG.VERSION')} [{get_database_metadata('EPSG.DATE')}]'), ('ESRI Database', f'{get_database_metadata('ESRI.VERSION')} [{get_database_metadata('ESRI.DATE')}]'), ('IGNF Database', f'{get_database_metadata('IGNF.VERSION')} [{get_database_metadata('IGNF.DATE')}]')]
    return dict(blob)
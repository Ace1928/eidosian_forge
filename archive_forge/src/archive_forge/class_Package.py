import os
from collections import namedtuple
import re
import sqlite3
import typing
import warnings
import rpy2.rinterface as rinterface
from rpy2.rinterface import StrSexpVector
from rpy2.robjects.packages_utils import (get_packagepath,
from collections import OrderedDict
class Package(object):
    """
    The R documentation page (aka help) for a package.
    """
    __package_path = None
    __package_name = None
    __aliases_info = 'aliases.rds'
    __hsearch_meta = os.path.join('Meta', 'hsearch.rds')
    __paths_info = 'paths.rds'
    __anindex_info = 'AnIndex'

    def __package_name_get(self):
        return self.__package_name
    name = property(__package_name_get, None, None, 'Name of the package as known by R')

    def __init__(self, package_name: str, package_path: typing.Optional[str]=None):
        self.__package_name = package_name
        if package_path is None:
            package_path = get_packagepath(package_name)
        self.__package_path = package_path
        rd_meta_dbcon = sqlite3.connect(':memory:')
        create_metaRd_db(rd_meta_dbcon)
        populate_metaRd_db(package_name, rd_meta_dbcon, package_path=package_path)
        self._dbcon = rd_meta_dbcon
        path = os.path.join(package_path, 'help', package_name + '.rdx')
        self._rdx = readRDS(StrSexpVector((path,)))

    def fetch(self, alias: str) -> Page:
        """ Fetch the documentation page associated with a given alias.

        For S4 classes, the class name is *often* suffixed with '-class'.
        For example, the alias to the documentation for the class
        AnnotatedDataFrame in the package Biobase is
        'AnnotatedDataFrame-class'.
        """
        c = self._dbcon.execute('SELECT rd_meta_rowid, alias FROM rd_alias_meta WHERE alias=?', (alias,))
        res_alias = c.fetchall()
        if len(res_alias) == 0:
            raise HelpNotFoundError('No help could be fetched', topic=alias, package=self.__package_name)
        c = self._dbcon.execute('SELECT file, name, type FROM rd_meta WHERE rowid=?', (res_alias[0][0],))
        res_all = c.fetchall()
        rkey = StrSexpVector((res_all[0][0][:-3],))
        _type = res_all[0][2]
        rpath = StrSexpVector((os.path.join(self.package_path, 'help', f'{self.__package_name}.rdb'),))
        rdx_variables = self._rdx[self._rdx.do_slot('names').index('variables')]
        _eval = rinterface.baseenv['eval']
        devnull_func = rinterface.parse('function(x) {}')
        devnull_func = _eval(devnull_func)
        res = _lazyload_dbfetch(rdx_variables[rdx_variables.do_slot('names').index(rkey[0])], rpath, self._rdx[self._rdx.do_slot('names').index('compressed')], devnull_func)
        p_res = Page(res, _type=_type)
        return p_res
    package_path = property(lambda self: str(self.__package_path), None, None, 'Path to the installed R package')

    def __repr__(self):
        r = 'R package %s %s' % (self.__package_name, super(Package, self).__repr__())
        return r
import os.path
import re
import sys
import shutil
from decimal import Decimal
from pyomo.common.dependencies import attempt_import
from pyomo.dataportal import TableData
from pyomo.dataportal.factory import DataManagerFactory
@DataManagerFactory.register('pyodbc', '%s database interface' % 'pyodbc')
class pyodbc_db_Table(db_Table):
    _drivers = {'mdb': ['Microsoft Access Driver (*.mdb)'], 'xls': ['Microsoft Excel Driver (*.xls, *.xlsx, *.xlsm, *.xlsb)', 'Microsoft Excel Driver (*.xls)'], 'xlsx': ['Microsoft Excel Driver (*.xls, *.xlsx, *.xlsm, *.xlsb)'], 'xlsm': ['Microsoft Excel Driver (*.xls, *.xlsx, *.xlsm, *.xlsb)'], 'xlsb': ['Microsoft Excel Driver (*.xls, *.xlsx, *.xlsm, *.xlsb)'], 'mysql': ['MySQL']}
    _drivers['access'] = _drivers['mdb']
    _drivers['excel'] = _drivers['xls']

    def __init__(self):
        db_Table.__init__(self)
        self.using = 'pyodbc'

    def available(self):
        return pyodbc_available

    def requirements(self):
        return 'pyodbc'

    def connect(self, connection, options):
        if not options.driver is None:
            ctype = options.driver
        elif '.' in connection:
            ctype = connection.split('.')[-1]
        elif 'mysql' in connection.lower():
            ctype = 'mysql'
        else:
            ctype = ''
        extras = {}
        if ctype in ['xls', 'xlsx', 'xlsm', 'xlsb', 'excel'] or '.xls' in connection or '.xlsx' in connection or ('.xlsm' in connection) or ('.xlsb' in connection):
            extras['autocommit'] = True
        connection = self.create_connection_string(ctype, connection, options)
        try:
            conn = db_Table.connect(self, connection, options, extras)
        except TypeError:
            raise
        except Exception:
            e = sys.exc_info()[1]
            code = e.args[0]
            if code == 'IM002' or code == '08001':
                if 'HOME' in os.environ:
                    odbcIniPath = os.path.join(os.environ['HOME'], '.odbc.ini')
                    if os.path.exists(odbcIniPath):
                        shutil.copy(odbcIniPath, odbcIniPath + '.orig')
                        config = ODBCConfig(filename=odbcIniPath)
                    else:
                        config = ODBCConfig()
                    dsninfo = self.create_dsn_dict(connection, config)
                    dsnid = re.sub('[^A-Za-z0-9]', '', dsninfo['Database'])
                    dsn = 'PYOMO{0}'.format(dsnid)
                    config.add_source(dsn, dsninfo['Driver'])
                    config.add_source_spec(dsn, dsninfo)
                    config.write(odbcIniPath)
                    connstr = 'DRIVER={{{0}}};DSN={1}'.format(dsninfo['Driver'], dsn)
                else:
                    config = ODBCConfig()
                    dsninfo = self.create_dsn_dict(connection, config)
                    connstr = []
                    for k, v in dsninfo.items():
                        if ' ' in v and (v[0] != '{' or v[-1] != '}'):
                            connstr.append('%s={%s}' % (k.upper(), v))
                        else:
                            connstr.append('%s=%s' % (k.upper(), v))
                    connstr = ';'.join(connstr)
                conn = db_Table.connect(self, connstr, options, extras)
            else:
                raise
        return conn

    def create_dsn_dict(self, argstr, existing_config):
        result = {}
        parts = argstr.split(';')
        argdict = {}
        for part in parts:
            if len(part) > 0 and '=' in part:
                key, val = part.split('=', 1)
                argdict[key.lower().strip()] = val.strip()
        if 'driver' in argdict:
            result['Driver'] = '{0}'.format(argdict['driver']).strip('{}')
        if 'dsn' in argdict:
            if argdict['dsn'] in existing_config.source_specs:
                return existing_config.source_specs[argdict['dsn']]
            else:
                import logging
                logger = logging.getLogger('pyomo.core')
                logger.warning('DSN with name {0} not found. Attempting to continue with options...'.format(argdict['dsn']))
        if 'dbq' in argdict:
            if 'Driver' not in result:
                result['Driver'] = self._drivers[argdict['dbq'].split('.')[-1].lower()]
            result['Database'] = argdict['dbq']
            result['Server'] = 'localhost'
            result['User'] = ''
            result['Password'] = ''
            result['Port'] = '5432'
            result['Description'] = argdict['dbq']
            for k in argdict.keys():
                if k.capitalize() not in result:
                    result[k.capitalize()] = argdict[k]
        elif 'Driver' not in result:
            raise Exception('No driver specified, and no DBQ to infer from')
        elif result['Driver'].lower() == 'mysql':
            result['Driver'] = 'MySQL'
            result['Server'] = argdict.get('server', 'localhost')
            result['Database'] = argdict.get('database', '')
            result['Port'] = argdict.get('port', '3306')
            result['Socket'] = argdict.get('socket', '')
            result['Option'] = argdict.get('option', '')
            result['Stmt'] = argdict.get('stmt', '')
            result['User'] = argdict.get('user', '')
            result['Password'] = argdict.get('password', '')
            result['Description'] = argdict.get('description', '')
        else:
            raise Exception("Unknown driver type '{0}' for database connection".format(result['Driver']))
        return result

    def create_connection_string(self, ctype, connection, options):
        driver = self._get_driver(ctype)
        if driver:
            if ' ' in driver and (driver[0] != '{' or driver[-1] != '}'):
                return 'DRIVER={%s};Dbq=%s;' % (driver, connection)
            else:
                return 'DRIVER=%s;Dbq=%s;' % (driver, connection)
        return connection

    def _get_driver(self, ctype):
        drivers = self._drivers.get(ctype, [])
        for driver in drivers:
            if driver in pyodbc.drivers():
                return driver
        if drivers:
            return drivers[0]
        else:
            return None
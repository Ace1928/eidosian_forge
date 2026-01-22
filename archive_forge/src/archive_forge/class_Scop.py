import os
import re
from urllib.parse import urlencode
from urllib.request import urlopen
from . import Des
from . import Cla
from . import Hie
from . import Residues
from Bio import SeqIO
from Bio.Seq import Seq
class Scop:
    """The entire SCOP hierarchy.

    root -- The root node of the hierarchy
    """

    def __init__(self, cla_handle=None, des_handle=None, hie_handle=None, dir_path=None, db_handle=None, version=None):
        """Build the SCOP hierarchy from the SCOP parsable files, or a sql backend.

        If no file handles are given, then a Scop object with a single
        empty root node is returned.

        If a directory and version are given (with dir_path=.., version=...) or
        file handles for each file, the whole scop tree will be built in memory.

        If a MySQLdb database handle is given, the tree will be built as needed,
        minimising construction times.  To build the SQL database to the methods
        write_xxx_sql to create the tables.

        """
        self._sidDict = {}
        self._sunidDict = {}
        if all((h is None for h in [cla_handle, des_handle, hie_handle, dir_path, db_handle])):
            return
        if dir_path is None and db_handle is None:
            if cla_handle is None or des_handle is None or hie_handle is None:
                raise RuntimeError('Need CLA, DES and HIE files to build SCOP')
        sunidDict = {}
        self.db_handle = db_handle
        try:
            if db_handle:
                pass
            else:
                if dir_path:
                    if not version:
                        raise RuntimeError('Need SCOP version to find parsable files in directory')
                    if cla_handle or des_handle or hie_handle:
                        raise RuntimeError('Cannot specify SCOP directory and specific files')
                    cla_handle = _open_scop_file(dir_path, version, 'cla')
                    des_handle = _open_scop_file(dir_path, version, 'des')
                    hie_handle = _open_scop_file(dir_path, version, 'hie')
                root = Node()
                domains = []
                root.sunid = 0
                root.type = 'ro'
                sunidDict[root.sunid] = root
                self.root = root
                root.description = 'SCOP Root'
                records = Des.parse(des_handle)
                for record in records:
                    if record.nodetype == 'px':
                        n = Domain()
                        n.sid = record.name
                        domains.append(n)
                    else:
                        n = Node()
                    n.sunid = record.sunid
                    n.type = record.nodetype
                    n.sccs = record.sccs
                    n.description = record.description
                    sunidDict[n.sunid] = n
                records = Hie.parse(hie_handle)
                for record in records:
                    if record.sunid not in sunidDict:
                        print(record.sunid)
                    n = sunidDict[record.sunid]
                    if record.parent != '':
                        if record.parent not in sunidDict:
                            raise ValueError('Incomplete data?')
                        n.parent = sunidDict[record.parent]
                    for c in record.children:
                        if c not in sunidDict:
                            raise ValueError('Incomplete data?')
                        n.children.append(sunidDict[c])
                sidDict = {}
                records = Cla.parse(cla_handle)
                for record in records:
                    n = sunidDict[record.sunid]
                    assert n.sccs == record.sccs
                    assert n.sid == record.sid
                    n.residues = record.residues
                    sidDict[n.sid] = n
                self._sunidDict = sunidDict
                self._sidDict = sidDict
                self._domains = tuple(domains)
        finally:
            if dir_path:
                if cla_handle:
                    cla_handle.close()
                if des_handle:
                    des_handle.close()
                if hie_handle:
                    hie_handle.close()

    def getRoot(self):
        """Get root node."""
        return self.getNodeBySunid(0)

    def getDomainBySid(self, sid):
        """Return a domain from its sid."""
        if sid in self._sidDict:
            return self._sidDict[sid]
        if self.db_handle:
            self.getDomainFromSQL(sid=sid)
            if sid in self._sidDict:
                return self._sidDict[sid]
        else:
            return None

    def getNodeBySunid(self, sunid):
        """Return a node from its sunid."""
        if sunid in self._sunidDict:
            return self._sunidDict[sunid]
        if self.db_handle:
            self.getDomainFromSQL(sunid=sunid)
            if sunid in self._sunidDict:
                return self._sunidDict[sunid]
        else:
            return None

    def getDomains(self):
        """Return an ordered tuple of all SCOP Domains."""
        if self.db_handle:
            return self.getRoot().getDescendents('px')
        else:
            return self._domains

    def write_hie(self, handle):
        """Build an HIE SCOP parsable file from this object."""
        for n in sorted(self._sunidDict.values(), key=lambda x: x.sunid):
            handle.write(str(n.toHieRecord()))

    def write_des(self, handle):
        """Build a DES SCOP parsable file from this object."""
        for n in sorted(self._sunidDict.values(), key=lambda x: x.sunid):
            if n != self.root:
                handle.write(str(n.toDesRecord()))

    def write_cla(self, handle):
        """Build a CLA SCOP parsable file from this object."""
        for n in sorted(self._sidDict.values(), key=lambda x: x.sunid):
            handle.write(str(n.toClaRecord()))

    def getDomainFromSQL(self, sunid=None, sid=None):
        """Load a node from the SQL backend using sunid or sid."""
        if sunid is None and sid is None:
            return None
        cur = self.db_handle.cursor()
        if sid:
            cur.execute('SELECT sunid FROM cla WHERE sid=%s', sid)
            res = cur.fetchone()
            if res is None:
                return None
            sunid = res[0]
        cur.execute('SELECT * FROM des WHERE sunid=%s', sunid)
        data = cur.fetchone()
        if data is not None:
            n = None
            if data[1] != 'px':
                n = Node(scop=self)
                cur.execute('SELECT child FROM hie WHERE parent=%s', sunid)
                children = []
                for c in cur.fetchall():
                    children.append(c[0])
                n.children = children
            else:
                n = Domain(scop=self)
                cur.execute('select sid, residues, pdbid from cla where sunid=%s', sunid)
                n.sid, n.residues, pdbid = cur.fetchone()
                n.residues = Residues.Residues(n.residues)
                n.residues.pdbid = pdbid
                self._sidDict[n.sid] = n
            n.sunid, n.type, n.sccs, n.description = data
            if data[1] != 'ro':
                cur.execute('SELECT parent FROM hie WHERE child=%s', sunid)
                n.parent = cur.fetchone()[0]
            n.sunid = int(n.sunid)
            self._sunidDict[n.sunid] = n

    def getAscendentFromSQL(self, node, type):
        """Get ascendents using SQL backend."""
        if nodeCodeOrder.index(type) >= nodeCodeOrder.index(node.type):
            return None
        cur = self.db_handle.cursor()
        cur.execute('SELECT ' + type + ' from cla WHERE ' + node.type + '=%s', node.sunid)
        result = cur.fetchone()
        if result is not None:
            return self.getNodeBySunid(result[0])
        else:
            return None

    def getDescendentsFromSQL(self, node, type):
        """Get descendents of a node using the database backend.

        This avoids repeated iteration of SQL calls and is therefore much
        quicker than repeatedly calling node.getChildren().
        """
        if nodeCodeOrder.index(type) <= nodeCodeOrder.index(node.type):
            return []
        des_list = []
        if node.type == 'ro':
            for c in node.getChildren():
                for d in self.getDescendentsFromSQL(c, type):
                    des_list.append(d)
            return des_list
        cur = self.db_handle.cursor()
        if type != 'px':
            cur.execute('SELECT DISTINCT des.sunid,des.type,des.sccs,description FROM cla,des WHERE cla.' + node.type + '=%s AND cla.' + type + '=des.sunid', node.sunid)
            data = cur.fetchall()
            for d in data:
                if int(d[0]) not in self._sunidDict:
                    n = Node(scop=self)
                    n.sunid, n.type, n.sccs, n.description = d
                    n.sunid = int(n.sunid)
                    self._sunidDict[n.sunid] = n
                    cur.execute('SELECT parent FROM hie WHERE child=%s', n.sunid)
                    n.parent = cur.fetchone()[0]
                    cur.execute('SELECT child FROM hie WHERE parent=%s', n.sunid)
                    children = []
                    for c in cur.fetchall():
                        children.append(c[0])
                    n.children = children
                des_list.append(self._sunidDict[int(d[0])])
        else:
            cur.execute('SELECT cla.sunid,sid,pdbid,residues,cla.sccs,type,description,sp FROM cla,des where cla.sunid=des.sunid and cla.' + node.type + '=%s', node.sunid)
            data = cur.fetchall()
            for d in data:
                if int(d[0]) not in self._sunidDict:
                    n = Domain(scop=self)
                    n.sunid, n.sid, pdbid, n.residues, n.sccs, n.type, n.description, n.parent = d[0:8]
                    n.residues = Residues.Residues(n.residues)
                    n.residues.pdbid = pdbid
                    n.sunid = int(n.sunid)
                    self._sunidDict[n.sunid] = n
                    self._sidDict[n.sid] = n
                des_list.append(self._sunidDict[int(d[0])])
        return des_list

    def write_hie_sql(self, handle):
        """Write HIE data to SQL database."""
        cur = handle.cursor()
        cur.execute('DROP TABLE IF EXISTS hie')
        cur.execute('CREATE TABLE hie (parent INT, child INT, PRIMARY KEY (child), INDEX (parent) )')
        for p in self._sunidDict.values():
            for c in p.children:
                cur.execute(f'INSERT INTO hie VALUES ({p.sunid},{c.sunid})')

    def write_cla_sql(self, handle):
        """Write CLA data to SQL database."""
        cur = handle.cursor()
        cur.execute('DROP TABLE IF EXISTS cla')
        cur.execute('CREATE TABLE cla (sunid INT, sid CHAR(8), pdbid CHAR(4), residues VARCHAR(50), sccs CHAR(10), cl INT, cf INT, sf INT, fa INT, dm INT, sp INT, px INT, PRIMARY KEY (sunid), INDEX (SID) )')
        for n in self._sidDict.values():
            c = n.toClaRecord()
            cur.execute('INSERT INTO cla VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)', (n.sunid, n.sid, c.residues.pdbid, c.residues, n.sccs, n.getAscendent('cl').sunid, n.getAscendent('cf').sunid, n.getAscendent('sf').sunid, n.getAscendent('fa').sunid, n.getAscendent('dm').sunid, n.getAscendent('sp').sunid, n.sunid))

    def write_des_sql(self, handle):
        """Write DES data to SQL database."""
        cur = handle.cursor()
        cur.execute('DROP TABLE IF EXISTS des')
        cur.execute('CREATE TABLE des (sunid INT, type CHAR(2), sccs CHAR(10), description VARCHAR(255), PRIMARY KEY (sunid) )')
        for n in self._sunidDict.values():
            cur.execute('INSERT INTO des VALUES (%s,%s,%s,%s)', (n.sunid, n.type, n.sccs, n.description))
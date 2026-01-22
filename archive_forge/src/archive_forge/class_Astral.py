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
class Astral:
    """Representation of the ASTRAL database.

    Abstraction of the ASTRAL database, which has sequences for all the SCOP domains,
    as well as clusterings by percent id or evalue.
    """

    def __init__(self, dir_path=None, version=None, scop=None, astral_file=None, db_handle=None):
        """Initialize the astral database.

        You must provide either a directory of SCOP files:
            - dir_path - string, the path to location of the scopseq-x.xx directory
                       (not the directory itself), and
            - version   -a version number.

        or, a FASTA file:
            - astral_file - string, a path to a fasta file (which will be loaded in memory)

        or, a MYSQL database:
            - db_handle - a database handle for a MYSQL database containing a table
              'astral' with the astral data in it.  This can be created
              using writeToSQL.

        """
        if astral_file is None and dir_path is None and (db_handle is None):
            raise RuntimeError('Need either file handle, or (dir_path + version), or database handle to construct Astral')
        if not scop:
            raise RuntimeError('Must provide a Scop instance to construct')
        self.scop = scop
        self.db_handle = db_handle
        if not astral_file and (not db_handle):
            if dir_path is None or version is None:
                raise RuntimeError('must provide dir_path and version')
            self.version = version
            self.path = os.path.join(dir_path, f'scopseq-{version}')
            astral_file = f'astral-scopdom-seqres-all-{self.version}.fa'
            astral_file = os.path.join(self.path, astral_file)
        if astral_file:
            self.fasta_dict = SeqIO.to_dict(SeqIO.parse(astral_file, 'fasta'))
        self.astral_file = astral_file
        self.EvDatasets = {}
        self.EvDatahash = {}
        self.IdDatasets = {}
        self.IdDatahash = {}

    def domainsClusteredByEv(self, id):
        """Get domains clustered by evalue."""
        if id not in self.EvDatasets:
            if self.db_handle:
                self.EvDatasets[id] = self.getAstralDomainsFromSQL(astralEv_to_sql[id])
            else:
                if not self.path:
                    raise RuntimeError('No scopseq directory specified')
                file_prefix = 'astral-scopdom-seqres-sel-gs'
                filename = '%s-e100m-%s-%s.id' % (file_prefix, astralEv_to_file[id], self.version)
                filename = os.path.join(self.path, filename)
                self.EvDatasets[id] = self.getAstralDomainsFromFile(filename)
        return self.EvDatasets[id]

    def domainsClusteredById(self, id):
        """Get domains clustered by percentage identity."""
        if id not in self.IdDatasets:
            if self.db_handle:
                self.IdDatasets[id] = self.getAstralDomainsFromSQL('id' + str(id))
            else:
                if not self.path:
                    raise RuntimeError('No scopseq directory specified')
                file_prefix = 'astral-scopdom-seqres-sel-gs'
                filename = f'{file_prefix}-bib-{id}-{self.version}.id'
                filename = os.path.join(self.path, filename)
                self.IdDatasets[id] = self.getAstralDomainsFromFile(filename)
        return self.IdDatasets[id]

    def getAstralDomainsFromFile(self, filename=None, file_handle=None):
        """Get the scop domains from a file containing a list of sids."""
        if file_handle is None and filename is None:
            raise RuntimeError('You must provide a filename or handle')
        if not file_handle:
            file_handle = open(filename)
        doms = []
        while True:
            line = file_handle.readline()
            if not line:
                break
            line = line.rstrip()
            doms.append(line)
        if filename:
            file_handle.close()
        doms = [a for a in doms if a[0] == 'd']
        doms = [self.scop.getDomainBySid(x) for x in doms]
        return doms

    def getAstralDomainsFromSQL(self, column):
        """Load ASTRAL domains from the MySQL database.

        Load a set of astral domains from a column in the astral table of a MYSQL
        database (which can be created with writeToSQL(...).
        """
        cur = self.db_handle.cursor()
        cur.execute('SELECT sid FROM astral WHERE ' + column + '=1')
        data = cur.fetchall()
        data = [self.scop.getDomainBySid(x[0]) for x in data]
        return data

    def getSeqBySid(self, domain):
        """Get the seq record of a given domain from its sid."""
        if self.db_handle is None:
            return self.fasta_dict[domain].seq
        else:
            cur = self.db_handle.cursor()
            cur.execute('SELECT seq FROM astral WHERE sid=%s', domain)
            return Seq(cur.fetchone()[0])

    def getSeq(self, domain):
        """Return seq associated with domain."""
        return self.getSeqBySid(domain.sid)

    def hashedDomainsById(self, id):
        """Get domains clustered by sequence identity in a dict."""
        if id not in self.IdDatahash:
            self.IdDatahash[id] = {}
            for d in self.domainsClusteredById(id):
                self.IdDatahash[id][d] = 1
        return self.IdDatahash[id]

    def hashedDomainsByEv(self, id):
        """Get domains clustered by evalue in a dict."""
        if id not in self.EvDatahash:
            self.EvDatahash[id] = {}
            for d in self.domainsClusteredByEv(id):
                self.EvDatahash[id][d] = 1
        return self.EvDatahash[id]

    def isDomainInId(self, dom, id):
        """Return true if the domain is in the astral clusters for percent ID."""
        return dom in self.hashedDomainsById(id)

    def isDomainInEv(self, dom, id):
        """Return true if the domain is in the ASTRAL clusters for evalues."""
        return dom in self.hashedDomainsByEv(id)

    def writeToSQL(self, db_handle):
        """Write the ASTRAL database to a MYSQL database."""
        cur = db_handle.cursor()
        cur.execute('DROP TABLE IF EXISTS astral')
        cur.execute('CREATE TABLE astral (sid CHAR(8), seq TEXT, PRIMARY KEY (sid))')
        for dom in self.fasta_dict:
            cur.execute('INSERT INTO astral (sid,seq) values (%s,%s)', (dom, self.fasta_dict[dom].seq))
        for i in astralBibIds:
            cur.execute('ALTER TABLE astral ADD (id' + str(i) + ' TINYINT)')
            for d in self.domainsClusteredById(i):
                cur.execute('UPDATE astral SET id' + str(i) + '=1  WHERE sid=%s', d.sid)
        for ev in astralEvs:
            cur.execute('ALTER TABLE astral ADD (' + astralEv_to_sql[ev] + ' TINYINT)')
            for d in self.domainsClusteredByEv(ev):
                cur.execute('UPDATE astral SET ' + astralEv_to_sql[ev] + '=1  WHERE sid=%s', d.sid)
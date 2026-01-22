from ._base import *
from .models import LazyDBCacheBase
class GSheetsDBCache(LazyDBCacheBase):

    def __init__(self, sheets_url=None, cache_name='LazyCacheDB', auth_path=None, *args, **kwargs):
        self.url = sheets_url
        self.auth = auth_path
        self.cache_name = cache_name
        self.indexes = {}
        self.setup_client()

    def setup_client(self):
        gspread = lazy_init('gspread')
        self.gc = gspread.service_account(filename=self.auth)
        if self.url:
            self.sheet = self.gc.open_by_url(self.url)
            logger.info(f'Loaded GSheetsDB from URL: {self.url}')
        else:
            try:
                self.sheet = self.gc.open(self.cache_name)
                logger.info(f'Loaded GSheetsDB from Name: {self.cache_name}')
            except Exception as e:
                self.sheet = None
                logger.error(f'Failed to Load GSheetsDB from Name: {self.cache_name}')
        if not self.sheet:
            self.sheet = self.gc.create(self.cache_name)
            logger.info(f'Created GSheetsDB by Name: {self.cache_name}')
            self.url = self.sheet.url
        self.refresh_index()

    def create_indexes(self, dbdata):
        for n, i in enumerate(dbdata):
            if i not in self.all_wks:
                idx = dbdata[i]
                wks = self.sheet.add_worksheet(i, index=n)
                wks.append_row(idx.schema_props)

    def dump_indexes(self, dbdata):
        for schema, idx in dbdata.items():
            wks = self.sheet.worksheet(schema)
            items = []
            for i in idx.index.values():
                d = i.dict()
                items.append(list(d.values()))
            wks.insert_rows(items, row=2)
        self.refresh_index()

    def refresh_index(self):
        self.index = self.all_wks_dict

    def get_header(self, wks):
        return wks.row_values(1)

    @property
    def all_wks(self):
        return list(self.sheet.worksheets())

    @property
    def all_wks_dict(self):
        return {n: s for n, s in enumerate(self.all_wks)}

    @property
    def cache_file(self):
        return None

    @property
    def cache_filepath(self):
        return None

    @property
    def exists(self):
        return bool(self.url)

    def dumps(self, data, *args, **kwargs):
        pass

    def loads(self, *args, **kwargs):
        pass

    def restore(self, *args, **kwargs):
        pass

    def save(self, db, *args, **kwargs):
        pass